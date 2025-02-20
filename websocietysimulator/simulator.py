import logging
import os
import json
from typing import Type, Any
from .tools import InteractionTool, CacheInteractionTool
from .tools.evaluation_tool import RecommendationEvaluator, SimulationEvaluator
from .agent.simulation_agent import SimulationAgent
from .llm import LLMBase
from .agent.recommendation_agent import RecommendationAgent
from .tasks.simulation_task import SimulationTask
from .tasks.recommendation_task import RecommendationTask
from tqdm import tqdm

logger = logging.getLogger("websocietysimulator")

class Simulator:
    def __init__(self, data_dir: str | None = None, block_set_dir: str | None = None, device: str = "auto", cache: bool = False, track: int = 2):
        """
        Initialize the Simulator.
        Args:
            data_dir: Path to the directory containing dataset files.
            block_set_dir: Path to the directory containing block set data files.
            device: Device to use for evaluation. "auto" (default) will use GPU if available, otherwise CPU. Available options: "gpu", "cpu", "auto".
            cache: Whether to use cache for interaction tool.
        """
        logger.info("Start initializing Simulator")
        self.data_dir = data_dir
        self.real_data_dir = block_set_dir
        if data_dir is None:
            self.interaction_tool = None
        else:
            if cache:
                logger.info("Using CacheInteractionTool")
                self.interaction_tool = CacheInteractionTool(data_dir, block_set_dir)
            else:
                logger.info("Using Normal InteractionTool")
                self.interaction_tool = InteractionTool(data_dir, block_set_dir)

        self.tasks = []  # List to store tasks
        self.groundtruth_data = []  # List to store groundtruth data
        self.agent_class = None
        self.llm = None
        self.track = track
        if track == 1:
            self.recommendation_evaluator = None
            self.simulation_evaluator = SimulationEvaluator(device)
        elif track == 2:
            self.recommendation_evaluator = RecommendationEvaluator()
            self.simulation_evaluator = None
        self.simulation_outputs = []
        self.evaluation_results = []
        logger.info("Simulator initialized")

    def set_interaction_tool(self, interaction_tool: InteractionTool | CacheInteractionTool):
        self.interaction_tool = interaction_tool

    def set_task_and_groundtruth(self, task_dir: str, groundtruth_dir: str, is_real_data: bool = False):
        """
        Load tasks from a directory.
        Args:
            task_dir: Directory containing task files.
            groundtruth_dir: Directory containing groundtruth files.
            is_real_data: Whether the data is real data. Default is False.
        """
        tasks, groundtruth_data = self._load_task_and_groundtruth(task_dir, groundtruth_dir)
        if is_real_data:
            self.real_tasks = tasks
            self.real_groundtruth_data = groundtruth_data
            logger.info(f"Loaded {len(tasks)} task-groundtruth pairs from real data")
        else:
            self.sim_tasks = tasks
            self.sim_groundtruth_data = groundtruth_data
            logger.info(f"Loaded {len(tasks)} task-groundtruth pairs from simulation data")

    def _load_task_and_groundtruth(self, task_dir: str, groundtruth_dir: str) -> tuple[list[SimulationTask | RecommendationTask], list[dict]]:
        tasks = []  # Clear previous tasks
        groundtruth_data = []

        # 获取所有task文件并按index排序
        task_files = sorted([f for f in os.listdir(task_dir) if f.startswith('task_') and f.endswith('.json')], 
                          key=lambda x: int(x.split('_')[1].split('.')[0]))
        task_indexs = [task_file.split('_')[1].split('.')[0] for task_file in task_files]
        task_cnt = max([int(index) for index in task_indexs]) + 1
        tasks = [None] * task_cnt
        groundtruth_data = [None] * task_cnt

        for task_file in task_files:
            # 获取对应的groundtruth文件
            task_index = task_file.split('_')[1].split('.')[0]
            groundtruth_file = f'groundtruth_{task_index}.json'
            groundtruth_path = os.path.join(groundtruth_dir, groundtruth_file)
            
            if not os.path.exists(groundtruth_path):
                logger.warning(f"Groundtruth file {groundtruth_file} not found for task {task_file}")
                continue

            # 读取task文件
            task_path = os.path.join(task_dir, task_file)
            with open(task_path, 'r') as f:
                task_data = json.load(f)
                task_type = task_data.get('type')

                # Determine scenario type and create corresponding object
                if task_type == 'user_behavior_simulation':
                    task = SimulationTask(
                        user_id=task_data['user_id'],
                        item_id=task_data['item_id']
                    )
                elif task_type == 'recommendation':
                    task = RecommendationTask(
                        user_id=task_data['user_id'],
                        candidate_category=task_data['candidate_category'],
                        candidate_list=task_data['candidate_list'],
                        loc=task_data['loc']
                    )
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")

            with open(groundtruth_path, 'r') as f:
                gt_data = json.load(f)
                
            # tasks.append(task)
            # groundtruth_data.append(gt_data)
            tasks[int(task_index)] = task
            groundtruth_data[int(task_index)] = gt_data

        return tasks, groundtruth_data

    def set_agent(self, agent_class: Type):
        """
        Set the agent class to be used for the simulation.
        Args:
            agent_class: A class inheriting from the abstract Agent class.
        """
        if not issubclass(agent_class, (SimulationAgent, RecommendationAgent)):
            raise ValueError("Agent class must inherit from SimulationAgent or RecommendationAgent.")
        self.agent_class = agent_class
        logger.info("Agent class set")

    def set_llm(self, llm: LLMBase | list[LLMBase]):
        """
        Set the LLM to be used for the simulation.
        Args:
            llm: A class inheriting from the abstract LLM class.
        """
        self.llm = llm
        logger.info("LLM set")

    def run_simulation(self, number_of_tasks: int | None = None, enable_threading: bool = False, max_workers: int | None = None, time_limitation: float | None = None, eval_res_output_file: str = "log/test.json", is_real_data: bool = False) -> list[Any]:
        """
        Run the simulation with optional multi-threading support and time limitation.
        
        Args:
            number_of_tasks: Number of tasks to run. If None, run all tasks.
            enable_threading: Whether to enable multi-threading. Default is False.
            max_workers: Maximum number of threads to use. If None, will use min(32, number_of_tasks).
            time_limitation: Time limit in minutes. If None, no time limit is applied.
        Returns:
            List of outputs from agents for each scenario.
        """
        if is_real_data:
            self.tasks = self.real_tasks
            logger.info("Running tasks with real data")
            self.real_simulation_outputs = self._run_simulation(number_of_tasks=number_of_tasks, enable_threading=enable_threading, max_workers=max_workers, time_limitation=time_limitation, eval_res_output_file=eval_res_output_file)
        else:
            self.tasks = self.sim_tasks
            logger.info("Running tasks with simulation data")
            self.sim_simulation_outputs = self._run_simulation(number_of_tasks=number_of_tasks, enable_threading=enable_threading, max_workers=max_workers, time_limitation=time_limitation, eval_res_output_file=eval_res_output_file)

    def _run_simulation(self, number_of_tasks: int | None = None, enable_threading: bool = False, max_workers: int | None = None, time_limitation: float | None = None, eval_res_output_file: str = "log/test.json") -> list[Any]:
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

        start_time = time.time()
        timeout_seconds = time_limitation * 60 if time_limitation else None

        logger.info("Running simulation")
        if not self.agent_class:
            raise RuntimeError("Agent class is not set. Use set_agent() to set it.")
        if not self.interaction_tool:
            raise RuntimeError("Interaction tool is not set. Use set_interaction_tool() to set it.")

        task_to_run = self.tasks[:number_of_tasks] if number_of_tasks is not None else self.tasks
        # 载入已完成的tasks
        if os.path.exists(eval_res_output_file):
            with open(eval_res_output_file, 'r') as f:
                done_outputs = json.load(f)
            if self.track == 1:
                done_outputs = [output for output in done_outputs if output is not None and output['output']['stars'] != 0]
            else:
                done_outputs = [output for output in done_outputs if output is not None and len(output['output']) > 1]
            done_tasks = [output['task'] for output in done_outputs]
        else:
            done_outputs = []
            done_tasks = []

        logger.info(f"Total tasks: {max(len(task_to_run) - len(done_tasks), 0)}")
        done_simulation_outputs = [None] * len(self.tasks)
        for idx, task in enumerate(self.tasks):  # 例如之前跑过400个，重新跑50个，则保留后面的350个tasks在done_simulation_outputs里，不影响outputs的保存
            if task is None:
                continue
            if task.to_dict() in done_tasks: 
                done_simulation_outputs[idx] = done_outputs[done_tasks.index(task.to_dict())]

        # 如果不启用多线程，使用原始的串行处理
        if not enable_threading:
            self.simulation_outputs = done_simulation_outputs
            for index, task in enumerate(task_to_run):
                if task is None:
                    continue
                if task.to_dict() in done_tasks:
                    continue
                # 检查是否超时
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    logger.warning(f"Time limit ({time_limitation} minutes) reached. Stopping simulation.")
                    break

                if isinstance(self.llm, list):
                    agent = self.agent_class(llm=self.llm[index%len(self.llm)])
                else:
                    agent = self.agent_class(llm=self.llm)
                agent.set_interaction_tool(self.interaction_tool)
                agent.insert_task(task)
                
                try:
                    output = agent.workflow()
                    result = {
                        "task": task.to_dict(),
                        "output": output
                    }
                except NotImplementedError:
                    result = {
                        "task": task.to_dict(),
                        "error": "Forward method not implemented by participant."
                    }
                self.simulation_outputs[index] = result
                with open(eval_res_output_file, 'w') as f:
                    json.dump(self.simulation_outputs, f, indent=4)
                logger.info(f"Simulation finished for task {index} - time: {time.time() - start_time:.2f}s")
        else:
            # 多线程处理
            from threading import Lock, Event
            
            log_lock = Lock()
            output_lock = Lock()
            cancel_event = Event()  # 添加取消事件标志
            self.simulation_outputs = done_simulation_outputs

            def process_task(task_index_tuple):
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                
                def run_agent_task(agent, task):
                    begin_time = time.time()
                    output = agent.workflow()
                    with log_lock:
                        logger.info(f"Task {index} finished - time: {time.time() - begin_time:.2f}s")
                    return output
                
                index, task = task_index_tuple
                if task is None:
                    return None
                if task.to_dict() in done_tasks:
                    return index
                # 检查是否已经被要求取消
                if cancel_event.is_set():
                    return None
                    
                if isinstance(self.llm, list):
                    agent = self.agent_class(llm=self.llm[index%len(self.llm)])
                else:
                    agent = self.agent_class(llm=self.llm)
                agent.set_interaction_tool(self.interaction_tool)
                agent.insert_task(task)
                
                try:
                    # 使用内部的ThreadPoolExecutor来执行单个任务，设置超时时间为5分钟
                    with ThreadPoolExecutor(max_workers=1) as single_task_executor:
                        future = single_task_executor.submit(run_agent_task, agent, task)
                        try:
                            output = future.result(timeout=300)  # 5 minutes timeout
                            result = {
                                "task": task.to_dict(),
                                "output": output
                            }
                        except TimeoutError:
                            logger.warning(f"Task {index} timed out")
                            # 强制关闭执行器
                            single_task_executor._threads.clear()
                            single_task_executor.shutdown(wait=False)
                            return index, None
                except NotImplementedError:
                    result = {
                        "task": task.to_dict(),
                        "error": "Forward method not implemented by participant."
                    }
                except Exception as e:
                    logger.error(f"Task {index} failed with error: {str(e)}")
                    return index, None
                
                with log_lock:
                    logger.info(f"Simulation finished for task {index}")
                with output_lock:
                    self.simulation_outputs[index] = result
                    with open(eval_res_output_file, 'w') as f:
                        json.dump(self.simulation_outputs, f, indent=4)
                return index

            # 确定线程数
            if max_workers is None:
                max_workers = min(32, len(task_to_run))
            else:
                max_workers = min(max_workers, len(task_to_run))
            max_workers = max(1, max_workers)
            
            logger.info(f"Running with {max_workers} threads")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(process_task, (i, task)): i 
                    for i, task in enumerate(task_to_run)
                }

                try:
                    for future in tqdm(as_completed(future_to_index, timeout=timeout_seconds)):
                        try:
                            index = future.result()
                        except Exception as e:
                            logger.error(f"Task failed with error: {str(e)}")
                except TimeoutError:
                    logger.error(f"Time limit ({time_limitation} minutes) reached.")
                    # 设置取消标志
                    cancel_event.set()
                    # 强制取消所有任务
                    for future in future_to_index:
                        future.cancel()
                    # 立即关闭执行器，不等待任务完成
                    executor._threads.clear()
                    executor.shutdown(wait=False)
                    raise TimeoutError

        logger.info("Simulation finished")
        logger.info("Total time: {:.2f}s".format(time.time() - start_time))
        # 过滤掉None值（未完成的任务）
        # 增加已完成的任务
        return self.simulation_outputs

    def evaluate(self, real_output_file: str | None = None, sim_output_file: str | None = None, eval_num_real_tasks: int | None = None, eval_num_sim_tasks: int | None = None) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        """
        Evaluate the simulation results using the loaded groundtruth data.
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating simulation results")
        real_simulation_outputs, real_groundtruth_data = self._load_output(real_output_file, self.real_groundtruth_data, eval_num_real_tasks)
        sim_simulation_outputs, sim_groundtruth_data = self._load_output(sim_output_file, self.sim_groundtruth_data, eval_num_sim_tasks)
        self.simulation_outputs = sim_simulation_outputs + real_simulation_outputs
        groundtruth_data = sim_groundtruth_data + real_groundtruth_data

        gt_none_count = sum([1 for item in groundtruth_data if item is None])
        sim_count = len(self.simulation_outputs)
        gt_count = len(self.real_groundtruth_data) + len(self.sim_groundtruth_data)
        self.sim_number = len(sim_simulation_outputs) - sum([1 for item in sim_groundtruth_data if item is None])

        evaluation_results = {}

        # 根据agent类型选择评估方法
        if issubclass(self.agent_class, RecommendationAgent):
            evaluation_results = self._evaluate_recommendation(groundtruth_data)
        elif issubclass(self.agent_class, SimulationAgent):
            evaluation_results, fine_grained_results = self._evaluate_simulation(groundtruth_data)

        # 添加数据条目信息到评估结果中
        evaluation_results['data_info'] = {
            'evaluated_count': sim_count - gt_none_count,
            'evaluated_simulation_count': len(sim_simulation_outputs) - gt_none_count,
            'evaluated_real_count': len(real_simulation_outputs),
            'original_simulation_count': sim_count - gt_none_count,
            'original_ground_truth_count': gt_count - gt_none_count
        }

        self.evaluation_results.append(evaluation_results)
        logger.info("Evaluation finished")
        if issubclass(self.agent_class, RecommendationAgent):
            return evaluation_results
        elif issubclass(self.agent_class, SimulationAgent):
            return evaluation_results, fine_grained_results

    def _load_output(self, output_file: str, groundtruth_data: list[dict], eval_num_tasks: int | None = None) -> tuple[list[dict], list[dict]]:
        if not os.path.exists(output_file):
            # raise RuntimeError("No simulation outputs to evaluate. Run simulation first.")
            logger.warning("No simulation outputs to evaluate. Run simulation first.")
            return [], []
        with open(output_file, 'r') as f:
            simulation_outputs = json.load(f)
        simulation_outputs = simulation_outputs[:eval_num_tasks] if eval_num_tasks is not None else simulation_outputs

        # 检查数据条目数量
        sim_count = len(simulation_outputs)
        gt_count = len(groundtruth_data)
        
        if sim_count != gt_count:
            logger.warning(f"Warning: Number of simulation outputs ({sim_count}) does not match ground truth data ({gt_count})")
            # 使用较小的数量
            eval_count = min(sim_count, gt_count)
            groundtruth_data = groundtruth_data[:eval_count]
            simulation_outputs = simulation_outputs[:eval_count]
        return simulation_outputs, groundtruth_data

    def _evaluate_recommendation(self, ground_truth_data: list[dict]) -> dict[str, Any]:
        """
        Evaluate recommendation results using groundtruth
        """
        # 从ground truth数据中提取真实POI
        # gt_pois = [item['ground truth'] for item in ground_truth_data]
        gt_pois = []
        
        pred_pois = []
        for output, gt_item in zip(self.simulation_outputs, ground_truth_data):
            if gt_item is None:
                continue
            gt_pois.append(gt_item['ground truth'])
            if output is not None:
                if isinstance(output['output'], dict):
                    pred_pois.append(output['output']['rating'])
                else:
                    pred_pois.append(output['output'])
            else:
                pred_pois.append([''])

        none_count = sum([1 for pois in pred_pois if pois == ['']])
        # 计算评估指标
        metrics = self.recommendation_evaluator.calculate_hr_at_n(
            ground_truth=gt_pois,
            predictions=pred_pois,
            number_sim=self.sim_number
        )

        return {
            'type': 'recommendation',
            'metrics': metrics.__dict__,
            'none_count': none_count
        }

    def _evaluate_simulation(self, ground_truth_data: list[dict]) -> dict[str, Any]:
        """
        Evaluate simulation results
        """
        simulated_data = []
        for output in self.simulation_outputs:
            if output is not None:
                simulated_data.append(output['output'])
            else:
                simulated_data.append({
                    'stars': 0,
                    'review': ''
                })
        none_count = sum([1 for data in simulated_data if data['stars'] == 0 and data['review'] == ''])
        metrics, fine_grained_metrics = self.simulation_evaluator.calculate_metrics(
            simulated_data=simulated_data,
            real_data=ground_truth_data,
            number_sim=self.sim_number
        )
        return {
            'type': 'simulation',
            'metrics': metrics.__dict__,
            'none_count': none_count
        }, fine_grained_metrics

    def get_evaluation_history(self) -> list[dict[str, Any]]:
        """
        Get the history of evaluation results
        Returns:
            List of evaluation results
        """
        return self.evaluation_results
