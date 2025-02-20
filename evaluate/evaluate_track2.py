from argparse import ArgumentParser
from websocietysimulator import Simulator
from websocietysimulator.llm import InfinigenceLLM
import os
import json

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-a", "--agent", type=str, default="DummyAgent_track2")
    parser.add_argument("-t", "--task_set", type=str, default="amazon", choices=["amazon", "goodreads", "yelp", "merged"])
    parser.add_argument("--num_real_tasks", type=int, default=20)
    parser.add_argument("--num_sim_tasks", type=int, default=20)
    parser.add_argument("--eval_num_real_tasks", type=int, default=20)
    parser.add_argument("--eval_num_sim_tasks", type=int, default=20)
    parser.add_argument("-m", "--multi_threading", action="store_true")
    parser.add_argument("-w", "--workers", type=int, default=32)
    parser.add_argument("-p", "--prefix", type=str, default="baseline")
    parser.add_argument("-s", "--suffix", type=str, default="real")
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    task_set = args.task_set
    num_real_tasks = args.num_real_tasks
    if num_real_tasks < 0:
        num_real_tasks = None
    num_sim_tasks = args.num_sim_tasks
    if num_sim_tasks < 0:
        num_sim_tasks = None
    eval_num_real_tasks = args.eval_num_real_tasks
    if eval_num_real_tasks < 0:
        eval_num_real_tasks = None
    eval_num_sim_tasks = args.eval_num_sim_tasks
    if eval_num_sim_tasks < 0:
        eval_num_sim_tasks = None
    prefix = args.prefix
    suffix = args.suffix

    # Initialize Simulator
    simulator = Simulator(data_dir="data", block_set_dir=f"tasks/track2/{task_set}-{suffix}", device="auto", cache=True, track=2)

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"tasks/track2/{task_set}/tasks", groundtruth_dir=f"tasks/track2/{task_set}/groundtruth", is_real_data=False)
    simulator.set_task_and_groundtruth(task_dir=f"tasks/track2/{task_set}-{suffix}/tasks", groundtruth_dir=f"tasks/track2/{task_set}-{suffix}/groundtruth", is_real_data=True)

    # Set the agent and LLM
    exec(f"from {args.agent} import MyRecommendationAgent")
    simulator.set_agent(eval("MyRecommendationAgent"))
    simulator.set_llm(InfinigenceLLM(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY")))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    sim_output_file = f'log/{prefix}_outputs_track2_{task_set}.json'
    real_output_file = f'log/{prefix}_outputs_track2_{task_set}-{suffix}.json'
    results_file = f'results/{prefix}_evaluation_results_track2_{task_set}.json'

    if not args.eval_only:
        simulator.run_simulation(number_of_tasks=num_sim_tasks, enable_threading=args.multi_threading, max_workers=args.workers, eval_res_output_file=sim_output_file, is_real_data=False, time_limitation=120)
        simulator.run_simulation(number_of_tasks=num_real_tasks, enable_threading=args.multi_threading, max_workers=args.workers, eval_res_output_file=real_output_file, is_real_data=True, time_limitation=120)

    # Evaluate the agent
    evaluation_results = simulator.evaluate(real_output_file, sim_output_file, eval_num_real_tasks, eval_num_sim_tasks)
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f'\nTask type: {evaluation_results["type"]}')
    print(f'none count = {evaluation_results["none_count"]}')
    print()
    metrics: dict[str, float | int] = evaluation_results["metrics"]
    metric_set = []
    for metric_name, metric_value in metrics.items():
        if metric_name.startswith("sim_"):
            if metric_name[4:] not in metric_set:
                metric_set.append(metric_name[4:])
        elif metric_name.startswith("real_"):
            if metric_name[5:] not in metric_set:
                metric_set.append(metric_name[5:])
        else:
            if metric_name not in metric_set:
                metric_set.append(metric_name)
    # output a table with the columns: metric_name, sim_value, real_value, value
    print(f'{"metric_name":^20} {"sim_value":^20} {"real_value":^20} {"value":^20}')
    for metric_name in metric_set:
        sim_value = metrics.get(f"sim_{metric_name}", None)
        real_value = metrics.get(f"real_{metric_name}", None)
        value = metrics.get(metric_name, None)
        assert value is not None, f"Value for metric {metric_name} is None"
        if isinstance(value, float):
            sim_value = f'{sim_value:.4f}' if sim_value is not None else '/'
            real_value = f'{real_value:.4f}' if real_value is not None else '/'
            value = f'{value:.4f}'
        else:
            sim_value = f'{sim_value}' if sim_value is not None else '/'
            real_value = f'{real_value}' if real_value is not None else '/'
            value = f'{value}'
        print(f'{metric_name:^20} {sim_value:^20} {real_value:^20} {value:^20}')
    
    top_1_hit_rate_value = metrics.get("top_1_hit_rate", None)
    top_3_hit_rate_value = metrics.get("top_3_hit_rate", None)
    top_5_hit_rate_value = metrics.get("top_5_hit_rate", None)
    # sim_average_hit_rate_value = metrics.get("sim_average_hit_rate", None)
    # real_average_hit_rate_value = metrics.get("real_average_hit_rate", None)
    average_hit_rate_value = metrics.get("average_hit_rate", None)
    print(f'{top_1_hit_rate_value:.6f},{top_3_hit_rate_value:.6f},{top_5_hit_rate_value:.6f},{average_hit_rate_value:.6f}')
