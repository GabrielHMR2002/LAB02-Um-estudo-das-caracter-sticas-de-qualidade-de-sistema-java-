import pandas as pd
import matplotlib.pyplot as plt


class GraphService:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        for col in self.df.columns:
            if col not in ["repository", "name", "full_name", "language", "created_at", "updated_at", "html_url"]:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df["created_at"] = pd.to_datetime(self.df["created_at"], errors="coerce")
        self.df["updated_at"] = pd.to_datetime(self.df["updated_at"], errors="coerce")

        self.metrics_cols = [
            'cbo', 'cboModified', 'fanin', 'fanout', 'wmc', 'dit', 'noc', 'rfc',
            'lcom', 'lcom*', 'tcc', 'lcc', 'totalMethodsQty', 'staticMethodsQty',
            'publicMethodsQty', 'privateMethodsQty', 'protectedMethodsQty',
            'defaultMethodsQty', 'visibleMethodsQty', 'abstractMethodsQty',
            'finalMethodsQty', 'synchronizedMethodsQty', 'totalFieldsQty',
            'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty',
            'protectedFieldsQty', 'defaultFieldsQty', 'finalFieldsQty',
            'synchronizedFieldsQty', 'nosi', 'loc', 'returnQty', 'loopQty',
            'comparisonsQty', 'tryCatchQty', 'parenthesizedExpsQty',
            'stringLiteralsQty', 'numbersQty', 'assignmentsQty',
            'mathOperationsQty', 'variablesQty', 'maxNestedBlocksQty',
            'anonymousClassesQty', 'innerClassesQty', 'lambdasQty',
            'uniqueWordsQty', 'modifiers', 'logStatementsQty'
        ]
        self.metrics_cols = [c for c in self.metrics_cols if c in self.df.columns]

    def plot_rq1_correlation(self, output_path: str = "rq1_heatmap.png"):
        data = self.df[["stars"] + self.metrics_cols]
        corr = data.corr(numeric_only=True)[["stars"]].sort_values(by="stars", ascending=False)

        plt.figure(figsize=(6, len(corr) / 3))
        plt.imshow(corr, cmap="coolwarm", aspect="auto")
        plt.colorbar(label="Correlation with Stars")
        plt.xticks([0], ["Stars"])
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation between Stars and CK Metrics")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[INFO] RQ1 heatmap saved at {output_path}")

    def plot_rq2_maturity(self, output_path: str = "rq2_scatter.png"):
        df = self.df.copy()
        df["created_at"] = df["created_at"].dt.tz_localize(None)
        df["age_years"] = ((pd.Timestamp.today() - df["created_at"]).dt.days / 365).round(1)
        df = df.dropna(subset=["age_years", "wmc", "lcom", "cbo"])

        metrics = ["wmc", "lcom", "cbo"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=False)

        for i, metric in enumerate(metrics):
            axes[i].scatter(df["age_years"], df[metric], alpha=0.6, s=20, color="steelblue")
            axes[i].set_xlabel("Repository Age (years)")
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f"Age × {metric.upper()}")
            axes[i].grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("RQ2: Relationship between Repository Age and Quality Metrics")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[INFO] RQ2 scatter plot saved at {output_path}")

    def plot_rq3_activity(self, output_path: str = "rq3_boxplot.png"):
        df = self.df.copy()
        now = pd.Timestamp.now(tz=df["updated_at"].dt.tz)
        df["days_since_update"] = (now - df["updated_at"]).dt.days

        # Adjusted to 180 days to actually represent "last 6 months"
        df["is_active"] = df["days_since_update"] <= 180

        metrics = ["wmc", "lcom", "cbo"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=False)

        for i, metric in enumerate(metrics):
            subset = df.dropna(subset=[metric, "is_active"])
            data = [
                subset.loc[subset["is_active"], metric],
                subset.loc[~subset["is_active"], metric],
            ]
            axes[i].boxplot(data, labels=["Active", "Inactive"], patch_artist=True)
            axes[i].set_title(f"{metric.upper()} vs Activity")
            axes[i].set_ylabel(metric.upper())

        plt.suptitle("RQ3: CK Metrics by Repository Activity (Last Update ≤ 6 months)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[INFO] RQ3 boxplot saved at {output_path}")

    def plot_rq4_size(self, output_path: str = "rq4_scatter.png"):
        df = self.df.copy()
        df = df.dropna(subset=["loc", "wmc", "lcom"])

        metrics = ["wmc", "lcom"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 6), sharey=False)

        for i, metric in enumerate(metrics):
            axes[i].scatter(df["loc"], df[metric], alpha=0.6, s=20, color="darkorange")
            axes[i].set_xlabel("Size (LOC)")
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f"LOC × {metric.upper()}")
            axes[i].grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("RQ4: Relationship between Size (LOC) and Quality Metrics")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[INFO] RQ4 scatter plot saved at {output_path}")
