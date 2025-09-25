import requests
import json
import time
import csv
from datetime import datetime

def get_top_java_repos(limit=1000):
    repos = []
    page = 1
    per_page = 100
    
    while len(repos) < limit:
        url = f"https://api.github.com/search/repositories"
        params = {
            'q': 'language:java',
            'sort': 'stars',
            'order': 'desc',
            'per_page': per_page,
            'page': page
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                for repo in items:
                    if len(repos) >= limit:
                        break
                    
                    repo_data = {
                        'name': repo['name'],
                        'full_name': repo['full_name'],
                        'owner': repo['owner']['login'],
                        'stars': repo['stargazers_count'],
                        'forks': repo['forks_count'],
                        'size': repo['size'],
                        'created_at': repo['created_at'],
                        'updated_at': repo['updated_at'],
                        'clone_url': repo['clone_url'],
                        'html_url': repo['html_url'],
                        'default_branch': repo['default_branch']
                    }
                    repos.append(repo_data)
                
                page += 1
                time.sleep(1)  
                
            elif response.status_code == 403:
                print("Rate limit exceeded. Waiting...")
                time.sleep(60)
            else:
                print(f"Error: {response.status_code}")
                break
                
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            time.sleep(5)
    
    return repos[:limit]

def calculate_age(created_at):
    created_date = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ')
    now = datetime.now()
    age_days = (now - created_date).days
    return round(age_days / 365.25, 2)

def get_releases_count(owner, repo_name):
    url = f"https://api.github.com/repos/{owner}/{repo_name}/releases"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            releases = response.json()
            return len(releases)
    except:
        pass
    return 0

def enrich_repo_data(repos):
    enriched_repos = []
    
    for i, repo in enumerate(repos):
        print(f"Processing repository {i+1}/{len(repos)}: {repo['full_name']}")
        
        repo['age_years'] = calculate_age(repo['created_at'])
        
        repo['releases'] = get_releases_count(repo['owner'], repo['name'])
        
        enriched_repos.append(repo)
        time.sleep(1)  
    
    return enriched_repos

def save_to_csv(repos, filename):
    if not repos:
        return
    
    fieldnames = repos[0].keys()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(repos)

def main():
    print("Coletando top 1000 repositórios Java...")
    repos = get_top_java_repos(50)
    
    print("Enriquecendo dados dos repositórios...")
    enriched_repos = enrich_repo_data(repos)
    
    print("Salvando dados...")
    save_to_csv(enriched_repos, 'top_1000_java_repos.csv')
    
    with open('top_1000_java_repos.json', 'w', encoding='utf-8') as f:
        json.dump(enriched_repos, f, indent=2, ensure_ascii=False)
    
    print("Coleta concluída!")
    print(f"Total de repositórios coletados: {len(enriched_repos)}")

if __name__ == "__main__":
    main()