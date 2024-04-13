import wandb


class WANDbAPIUtils:

    def __init__(self, project_name):
        self.api = wandb.Api()
        self.runs = self.api.runs(path=project_name)

    def tag_best_rank(self, exp_id):
        runs = self.runs

        # Get runs with the same experiment id and full reconstruction
        matching_runs = [run for run in runs if run.config.get("exp_id") == exp_id and run.config.get("full_reconstruction") == True]

        # get list of rank
        ranks = [run.config.get("rank") for run in matching_runs]
        # get lowest rank index 
        min_rank_idx = ranks.index(min(ranks))

        # add tags to the best run
        matching_runs[min_rank_idx].tags = ["lowest_rank"]
        print(f"Best run {matching_runs[min_rank_idx].name} has rank {ranks[min_rank_idx]}")
        matching_runs[min_rank_idx].update()

    def update_all_best_rank_tags(self):
        runs = self.runs

        # Get runs with full reconstruction and experiment id
        matching_runs = [run for run in runs if run.config.get("full_reconstruction") == True and run.config.get("exp_id") != None]

        # group runs by experiment id
        exp_id_to_runs = {}
        for run in matching_runs:
            exp_id = run.config.get("exp_id")
            if exp_id not in exp_id_to_runs:
                exp_id_to_runs[exp_id] = []
            exp_id_to_runs[exp_id].append(run)

        for exp, runs in exp_id_to_runs.items():
            print(f"Experiment {exp} has {len(runs)} runs with full reconstruction")
            # get list of rank
            ranks = [run.config.get("rank") for run in runs]
            # get lowest rank index 
            min_rank_idx = ranks.index(min(ranks))

            # add tags to the best run
            runs[min_rank_idx].tags = ["lowest_rank"]
            print(f"Best run {runs[min_rank_idx].name} has rank {ranks[min_rank_idx]}")
            runs[min_rank_idx].update()

if __name__ == '__main__':
    project_name = "GraphEmbeddings"  # format: "username/projectname"
    exp_id = "2533580a-b17b-452f-ad0c-aaf36d733af2"  # experiment id to tag

    wandb_api = WANDbAPIUtils(project_name)
    # wandb_api.tag_best_rank(exp_id)

    wandb_api.update_all_best_rank_tags()
