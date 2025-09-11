import os
import subprocess
import csv
import json
import shutil
import sys
from pathlib import Path

class JavaMetricsCollector:
    def __init__(self, repos_file="top_1000_java_repos.csv", ck_jar_path="ck.jar"):
        self.repos_file = repos_file
        self.ck_jar_path = ck_jar_path
        self.base_dir = Path("repositories")
        self.metrics_dir = Path("metrics")
        self.results_dir = Path("results")
        
        self.base_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def load_repositories(self):
        repos = []
        with open(self.repos_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            repos = list(reader)
        return repos
    
    def clone_repository(self, repo_data):
        repo_name = repo_data['full_name']
        clone_url = repo_data['clone_url']
        repo_path = self.base_dir / repo_name.replace('/', '_')
        
        if repo_path.exists():
            print(f"Repository {repo_name} already exists, skipping...")
            return repo_path
        
        try:
            print(f"Cloning {repo_name}...")
            result = subprocess.run([
                'git', 'clone', '--depth', '1', clone_url, str(repo_path)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"Successfully cloned {repo_name}")
                return repo_path
            else:
                print(f"Failed to clone {repo_name}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"Timeout cloning {repo_name}")
            return None
        except Exception as e:
            print(f"Error cloning {repo_name}: {e}")
            return None
    
    def run_ck_analysis(self, repo_path, repo_name):
        if not repo_path.exists():
            return None
        
        output_dir = self.metrics_dir / repo_name.replace('/', '_')
        output_dir.mkdir(exist_ok=True)
        
        try:
            print(f"Running CK analysis on {repo_name}...")
            cmd = [
                'java', '-jar', self.ck_jar_path,
                str(repo_path),
                'true', '0', 'false',
                str(output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"CK analysis completed for {repo_name}")
                return output_dir
            else:
                print(f"CK analysis failed for {repo_name}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"CK analysis timeout for {repo_name}")
            return None
        except Exception as e:
            print(f"Error running CK analysis on {repo_name}: {e}")
            return None
    
    def count_loc_and_comments(self, repo_path):
        if not repo_path.exists():
            return 0, 0
        
        loc = 0
        comment_lines = 0
        
        try:
            for java_file in repo_path.rglob("*.java"):
                try:
                    with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                        in_block_comment = False
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            loc += 1
                            
                            if line.startswith('//'):
                                comment_lines += 1
                            elif '/*' in line and '*/' in line:
                                comment_lines += 1
                            elif '/*' in line:
                                in_block_comment = True
                                comment_lines += 1
                            elif in_block_comment:
                                comment_lines += 1
                                if '*/' in line:
                                    in_block_comment = False
                                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error counting LOC for {repo_path}: {e}")
        
        return loc, comment_lines
    
    def summarize_ck_metrics(self, metrics_dir):
        if not metrics_dir.exists():
            return {}
        
        summary = {
            'cbo_mean': 0, 'cbo_median': 0, 'cbo_std': 0,
            'dit_mean': 0, 'dit_median': 0, 'dit_std': 0,
            'lcom_mean': 0, 'lcom_median': 0, 'lcom_std': 0,
            'total_classes': 0
        }
        
        try:
            class_csv = metrics_dir / "class.csv"
            if class_csv.exists():
                import pandas as pd
                
                df = pd.read_csv(class_csv)
                if not df.empty:
                    if 'cbo' in df.columns:
                        summary['cbo_mean'] = df['cbo'].mean()
                        summary['cbo_median'] = df['cbo'].median()
                        summary['cbo_std'] = df['cbo'].std()
                    
                    if 'dit' in df.columns:
                        summary['dit_mean'] = df['dit'].mean()
                        summary['dit_median'] = df['dit'].median()
                        summary['dit_std'] = df['dit'].std()
                    
                    if 'lcom' in df.columns:
                        summary['lcom_mean'] = df['lcom'].mean()
                        summary['lcom_median'] = df['lcom'].median()
                        summary['lcom_std'] = df['lcom'].std()
                    
                    summary['total_classes'] = len(df)
                    
        except Exception as e:
            print(f"Error summarizing metrics: {e}")
        
        return summary
    
    def process_single_repository(self, repo_data):
        repo_name = repo_data['full_name']
        print(f"\n=== Processing {repo_name} ===")
        
        repo_path = self.clone_repository(repo_data)
        if not repo_path:
            return None
        
        loc, comments = self.count_loc_and_comments(repo_path)
        
        metrics_dir = self.run_ck_analysis(repo_path, repo_name)
        if not metrics_dir:
            return None
        
        ck_summary = self.summarize_ck_metrics(metrics_dir)
        
        result = {
            'repository': repo_name,
            'stars': int(repo_data['stars']),
            'age_years': float(repo_data['age_years']),
            'releases': int(repo_data['releases']),
            'loc': loc,
            'comments': comments,
            **ck_summary
        }
        
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)
        
        return result
    
    def process_repositories(self, max_repos=None):
        repos = self.load_repositories()
        if max_repos:
            repos = repos[:max_repos]
        
        results = []
        
        for i, repo in enumerate(repos, 1):
            print(f"\nProcessing {i}/{len(repos)}")
            result = self.process_single_repository(repo)
            if result:
                results.append(result)
                
                self.save_results(results, f"partial_results_{i}.csv")
            
            if i % 10 == 0:
                print(f"Completed {i} repositories")
        
        return results
    
    def save_results(self, results, filename="java_metrics_results.csv"):
        if not results:
            return
        
        output_file = self.results_dir / filename
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to {output_file}")

def main():
    if len(sys.argv) > 1:
        max_repos = int(sys.argv[1])
    else:
        max_repos = 1 
    
    collector = JavaMetricsCollector()
    
    print("Checking dependencies...")
    if not os.path.exists(collector.ck_jar_path):
        print(f"CK jar not found at {collector.ck_jar_path}")
        print("Please download CK from: https://github.com/mauricioaniche/ck/releases")
        return
    
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed. Please run: pip install pandas")
        return
    
    print(f"Processing {max_repos} repositories...")
    results = collector.process_repositories(max_repos)
    
    if results:
        collector.save_results(results)
        print(f"\nCompleted! Processed {len(results)} repositories successfully.")
    else:
        print("No repositories were processed successfully.")

if __name__ == "__main__":
    main()