import time
import json
import argparse
import numpy as np
import os
from tqdm import tqdm

from players import MatrixChatPlayer
from games.negotiation import NegotiationGame
from analysis import analyze
import wandb
import random

from matrix import Cli

def simulate_trials(
    model_id_1, model_id_2, output_dir, temperature, objective="self", num_runs=1, selfplay=True, 
    use_wandb=False, wandb_project=None
):
    # this function simulates num_runs trials of the game

    # sample from possible game contexts
    with open("data/selfplay.txt", "r") as file:
        # read lines into a list, stripping the newline character from each line
        lines = [line.strip() for line in file]

    np.random.seed(42)
    random_indices = np.random.randint(0, 4085, size=num_runs)

    p0_outcomes = []
    p1_outcomes = []

    # create log directories if starting a completely new run
    if not os.path.exists(f"{output_dir}/json_logs"):
        os.makedirs(f"{output_dir}/json_logs")

    if not os.path.exists(f"{output_dir}/text_logs"):
        os.makedirs(f"{output_dir}/text_logs")

    if not os.path.exists(f"{output_dir}/results"):
        os.makedirs(f"{output_dir}/results")

    if objective == "self":
        prompt_path = "prompts/dond.txt"
    elif objective == "coop":
        prompt_path = "prompts/coop_dond.txt"
    else:
        prompt_path = "prompts/comp_dond.txt"

    model_name_1 = model_id_1
    model_name_2 = model_id_2
    metadata_1 = Cli().get_app_metadata(app_name=model_name_1)
    metadata_2 = Cli().get_app_metadata(app_name=model_name_2)

    for i in tqdm(range(0, num_runs)):
        # initialize log files for trial i
        index = ("000" + str(i))[-3:]

        # full log
        game_filename = f"{output_dir}/text_logs/" + index + "_full.txt"
        if os.path.exists(game_filename):
            continue
        open(game_filename, "x")

        # # initialize a log file for each player
        p0_filename = f"{output_dir}/text_logs/{index}_p0.txt"
        p1_filename = f"{output_dir}/text_logs/{index}_p1.txt"
        open(p0_filename, "x")
        open(p1_filename, "x")

        open(f"{output_dir}/json_logs/{index}_p0.json", "x")
        open(f"{output_dir}/json_logs/{index}_p1.json", "x")

        # configure the game with randomly chosen item counts and values
        cnts, p1_vals = parse_context(lines[random_indices[i] * 2])
        _, p2_vals = parse_context(lines[random_indices[i] * 2 + 1])

        keys = ["book", "hat", "ball"]

        # initialize the game with the chosen configurations
        game = NegotiationGame(
            game_index=index,
            item_counts={keys[i]: cnts[i] for i in range(3)},
            p1_values={keys[i]: p1_vals[i] for i in range(3)},
            p2_values={keys[i]: p2_vals[i] for i in range(3)},
            game_log_filename=game_filename,
            objective=objective,
        )
        # print(game_filename)

        # initialize the players, depending on whether we are doing selfplay or have a human in the loop
        if random.random() < 0.5:
            # swap models half the time to ensure fairness
            m1, m2 = model_name_1, model_name_2
            meta1, meta2 = metadata_1, metadata_2
        else:
            m1, m2 = model_name_2, model_name_1
            meta1, meta2 = metadata_2, metadata_1

        players = [
            MatrixChatPlayer(
                game,
                game.player_values[0],
                log_filename=p0_filename,
                temperature=temperature,
                model=m1,
                prompt_path=prompt_path,
                metadata=meta1
            ),
            MatrixChatPlayer(
                game,
                game.player_values[1],
                log_filename=p1_filename,
                temperature=temperature,
                model=m2,
                prompt_path=prompt_path,
                metadata=meta2
            ),
        ]
        
        # play the game
        game_outcome = game.play_game(players)

        # write scores to score files
        with open(f"{output_dir}/p0_scores", "a") as p0_scores, open(
            f"{output_dir}/p1_scores", "a"
        ) as p1_scores:
            p0_scores.write(f"{game_outcome['p0_score']} \n")
            p1_scores.write(f"{game_outcome['p1_score']} \n")

        # write logs to JSON log files
        with open(f"{output_dir}/json_logs/{index}_p0.json", "w") as p0_log_file, open(
            f"{output_dir}/json_logs/{index}_p1.json", "w"
        ) as p1_log_file:
            json.dump(game_outcome["p0_log"], p0_log_file)
            json.dump(game_outcome["p1_log"], p1_log_file)

        # log the game context
        summary = {
            "counts": game.item_counts,
            "p0_values": game.player_values[0],
            "p1_values": game.player_values[1],
            "p0_allocation": game.proposals[0],
            "p1_allocation": game.proposals[1],
            "p0_score": game_outcome["p0_score"],
            "p1_score": game_outcome["p1_score"],
            "message_count": game_outcome["msg_cnt"],
            "token_count": game_outcome["token_cnt"],
            "is_valid_deal": game_outcome["is_valid_deal"],
        }
        with open(f"{output_dir}/results/{index}.json", "w") as summary_log:
            json.dump(summary, summary_log)

        # append to the list of simulated player outcomes
        p0_outcomes.append(game_outcome["p0_score"])
        p1_outcomes.append(game_outcome["p1_score"])

    # evaluate player outcomes
    print("p0 mean:", np.mean(p0_outcomes))
    print("p1 mean:", np.mean(p1_outcomes))
    
    # Log metrics to wandb if enabled
    if use_wandb:
        # Get detailed analysis metrics
        analysis_results = analyze(output_dir, verbose=False, objective=objective)
        
        # Create metrics dictionary similar to create_csv format
        metrics = {
            "metrics/p0_mean": np.mean(p0_outcomes),
            "metrics/p1_mean": np.mean(p1_outcomes),
            "metrics/total_mean": analysis_results["total"]["mean"],
            "metrics/total_median": analysis_results["total"]["median"],
            "metrics/total_avg_length_in_tokens": analysis_results["total"]["length_in_tkns"],
            "metrics/total_avg_length_in_msgs": analysis_results["total"]["length_in_msgs"],
            "metrics/abort_rate": analysis_results["total"]["abort_rate"],
            "metrics/agreement_proportion": analysis_results["agreement"]["proportion_agreement"],
            "metrics/pareto_optimal_proportion": analysis_results["agreement"]["proportion_pareto_opt"],
            "metrics/agreement_mean": analysis_results["agreement"]["mean"],
            "metrics/agreement_median": analysis_results["agreement"]["median"],
            "metrics/agreement_avg_length_in_tokens": analysis_results["agreement"]["length_in_tkns"],
            "metrics/agreement_avg_length_in_msgs": analysis_results["agreement"]["length_in_msgs"],
            "metrics/above_avg_mean": analysis_results["above_avg"]["mean"],
            "metrics/above_avg_median": analysis_results["above_avg"]["median"],
            "metrics/above_avg_avg_length_in_tokens": analysis_results["above_avg"]["length_in_tkns"],
            "metrics/above_avg_avg_length_in_msgs": analysis_results["above_avg"]["length_in_msgs"],
            "metrics/proportion_above_avg": analysis_results["above_avg"]["proportion_above_avg"],
        }
        
        # Log all metrics to wandb
        wandb.log(metrics)


def parse_context(ctx):
    # ctx is the string
    ctx = ctx.split()
    cnts = [int(n) for n in ctx[0::2]]
    vals = [int(v) for v in ctx[1::2]]
    return cnts, vals


def main():
    # parse CLI for objective and for model
    parser = argparse.ArgumentParser(
        description="A script that demonstrates argparse usage"
    )

    parser.add_argument(
        "-o", "--objective", type=str, default="self", help="Game objective"
    )
    parser.add_argument("-m1", "--model1", type=str, help="Model ID for first player")
    parser.add_argument("-m2", "--model2", type=str, help="Model ID for second player")
    parser.add_argument("-t", "--temp", type=float, help="Temperature")
    parser.add_argument("-n", "--num_runs", type=int, help="Number of self-play games")
    parser.add_argument("-d", "--output_dir", type=str, help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, help="Wandb project name", default='negotiation-eval')

    # parse the command-line arguments
    args = parser.parse_args()

    objective = args.objective
    model_id_1 = args.model1
    if args.model2 is None:
        model_id_2 = model_id_1
    else:
        model_id_2 = args.model2

    output_dir = args.output_dir
    temperature = args.temp
    num_runs = args.num_runs
    use_wandb = args.wandb
    wandb_project = args.wandb_project

    # Initialize wandb if enabled
    if use_wandb:
        # Create run_name from output_dir, removing "data/" prefix if present
        run_name = output_dir
        if run_name.startswith("data/"):
            run_name = run_name[5:]  # Remove "data/" prefix
        
        wandb.init(
            project=wandb_project or "lm-selfplay",
            name=run_name,
            config={
                "args/model_id_1": model_id_1,
                "args/model_id_2": model_id_2,
                "args/temperature": temperature,
                "args/objective": objective,
                "args/num_runs": num_runs,
                "args/output_dir": output_dir,
                "args/selfplay": True
            }
        )

    # duration calculation
    start_time = time.time()
    simulate_trials(model_id_1, model_id_2, output_dir, temperature, objective, num_runs, 
                   selfplay=True, use_wandb=use_wandb, wandb_project=wandb_project)
    end_time = time.time()

    duration = end_time - start_time 
    print(f"The function took {duration} seconds to complete.")
    
    if use_wandb:
        wandb.log({"metrics/duration_seconds": duration})
        wandb.finish()


if __name__ == "__main__":
    main()

