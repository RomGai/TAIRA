import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from agents.interact_agent import InteractorAgent
from agents.item_retrieval_agent import ItemRetrievalAgent
from agents.searcher_agent import SearcherAgent
from agents.task_interpreter_agent import InterpreterAgent
from core.manager_core import TAIRAManager
from user_simulate.evaluate_agent import EvaluateAgent
from utils.memory import Memory


pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.unicode.east_asian_width', True)


def setup_logger(log_file):
    """Setup logger for each query."""
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    f_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger


def load_config(config_path='system_config.yaml'):
    with open(config_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run TAIRA either with manager scheduling or a fixed pipeline.'
    )
    parser.add_argument(
        '--config',
        default='system_config.yaml',
        help='Path to the YAML config file.'
    )
    parser.add_argument(
        '--data-dir',
        help='Dataset directory under data/, e.g. data/amazon_music. Overrides config DOMAIN.'
    )
    parser.add_argument(
        '--query-file',
        default='query_data1.csv',
        help='Query CSV filename inside the dataset directory.'
    )
    parser.add_argument(
        '--execution-mode',
        choices=['manager', 'pipeline'],
        default='manager',
        help='manager = current planner/agent scheduling, pipeline = fixed ordered pipeline.'
    )
    parser.add_argument(
        '--pipeline',
        default='retrieve,interact',
        help='Comma-separated pipeline used in pipeline mode. Supported steps: search,retrieve,interact.'
    )
    parser.add_argument(
        '--query-number',
        type=int,
        help='Optional override for QUERY_NUMBER.'
    )
    parser.add_argument(
        '--classification-only',
        action='store_true',
        help='Only run rows where classification == 1.'
    )
    return parser.parse_args()


def resolve_dataset(config, data_dir=None):
    if data_dir:
        dataset_path = Path(data_dir)
        if not dataset_path.is_absolute():
            dataset_path = Path(data_dir)
        config['DOMAIN'] = dataset_path.name
        return config['DOMAIN'], dataset_path

    domain = config['DOMAIN']
    return domain, Path('data') / domain


def build_target_product(row, domain):
    if domain in ['amazon_clothing', 'amazon_music']:
        return f"{row['title']} | {row['category']}"
    if domain == 'amazon_beauty':
        return f"{row['title']} | {row['description']} | {row['category']}"
    return 'no target'


def init_agents(memory):
    item_agent = ItemRetrievalAgent(memory)
    searcher_agent = SearcherAgent(memory)
    interactor_agent = InteractorAgent(memory)
    interpreter = InterpreterAgent(memory)
    return item_agent, searcher_agent, interactor_agent, interpreter


def run_manager_mode(memory, row, domain, config, logger, agents):
    item_agent, searcher_agent, interactor_agent, interpreter = agents
    manager = TAIRAManager(
        memory,
        row['new_query'],
        build_target_product(row, domain),
        row['targets'],
        row['target_count'],
        row['preferences'],
        config,
        logger=logger,
    )

    manager.register_agent(item_agent)
    manager.register_agent(searcher_agent)
    manager.register_agent(interactor_agent)
    manager.register_agent(interpreter)

    return manager.delegate_task()


def run_pipeline_mode(memory, row, domain, config, logger, agents, pipeline_steps):
    item_agent, searcher_agent, interactor_agent, _ = agents
    evaluator = EvaluateAgent(memory, logger, config)

    user_input = row['new_query']
    target_product = build_target_product(row, domain)
    targets = row['targets']
    target_count = row['target_count']
    preference = row['preferences']

    memory.add_input(user_input)
    current_query = user_input
    outputs = {}

    for step in pipeline_steps:
        if step == 'search':
            current_query = str(searcher_agent.execute_task(current_query))
            memory.add_observation('SearcherAgent', user_input, current_query)
            outputs['search'] = current_query
            logger.debug('Pipeline step SearcherAgent output: %s', current_query)
        elif step == 'retrieve':
            retrieval_df = item_agent.execute_task(current_query)
            retrieval_records = retrieval_df.to_dict(orient='records')
            outputs['retrieve'] = retrieval_records
            memory.add_observation('ItemRetrievalAgent', current_query, retrieval_records)
            logger.debug('Pipeline step ItemRetrievalAgent output: %s', json.dumps(retrieval_records, ensure_ascii=False))
        elif step == 'interact':
            instruction = f'Generate recommendations from the existing task history for query: {user_input}'
            final_response = interactor_agent.generate_response(instruction)
            outputs['interact'] = final_response
            memory.add_observation('InteractorAgent', instruction, final_response)
            logger.debug('Pipeline step InteractorAgent output: %s', final_response)
        else:
            raise ValueError(f'Unsupported pipeline step: {step}')

    final_response = outputs.get('interact')
    if not final_response:
        retrieval_records = outputs.get('retrieve', [])
        recommendations = [{
            'recommendation': 'direct pipeline',
            'items': [
                {'id': str(item['product_id']), 'title': item['project_info']}
                for item in retrieval_records[:10]
            ],
        }]
        final_json = {'recommendations': recommendations}
    else:
        import re
        match = re.search(r'\{.*\}', final_response, re.DOTALL)
        if not match:
            raise ValueError('InteractorAgent did not return valid JSON content.')
        final_json = json.loads(match.group(0))

    hit_rate, mrr, ndcg, fail_flag = evaluator.evaluate(
        user_input,
        final_json,
        target_product,
        targets,
        target_count,
        preference=preference,
    )
    return hit_rate, mrr, ndcg, fail_flag, 'direct_pipeline'


def process_queries(df, domain, dataset_path, config, execution_mode, pipeline_steps):
    method = config['METHOD'] if execution_mode == 'manager' else f"pipeline-{'-'.join(pipeline_steps)}"
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H_%M_%S')
    log_dir = dataset_path / 'logs' / f'{method}-{formatted_time}'
    log_dir.mkdir(parents=True, exist_ok=True)
    results_csv = log_dir / f'result-{method}-{formatted_time}.csv'

    memory = Memory()
    agents = init_agents(memory)

    for index, row in df.iterrows():
        log_file = log_dir / f'log_{index + 1}.log'
        logger = setup_logger(str(log_file))
        print(f'Processing query {index + 1}')

        try:
            if execution_mode == 'manager':
                hit_rate, mrr, ndcg, fail_flag, pattern_key = run_manager_mode(
                    memory, row, domain, config, logger, agents
                )
            else:
                hit_rate, mrr, ndcg, fail_flag, pattern_key = run_pipeline_mode(
                    memory, row, domain, config, logger, agents, pipeline_steps
                )

            row['hit_rate'] = hit_rate
            row['mrr'] = mrr
            row['ndcgs'] = ndcg
            row['fail'] = 1 if fail_flag else 0
            row['pattern_used'] = pattern_key
        except Exception as exc:
            error_msg = f'Error processing query {index + 1}: {exc}'
            print(error_msg)
            logger.error(error_msg)
            row['hit_rate'] = 0
            row['mrr'] = 0
            row['ndcgs'] = 0
            row['fail'] = 1
            row['pattern_used'] = 'error'

        row_df = pd.DataFrame([row])
        if not results_csv.exists():
            row_df.to_csv(results_csv, mode='w', header=True, index=False)
        else:
            row_df.to_csv(results_csv, mode='a', header=False, index=False)

        memory.remove_data()
        logger.handlers.clear()

    complete_df = pd.read_csv(results_csv, encoding='ISO-8859-1')
    mean_row = pd.DataFrame({
        'hit_rate': [complete_df['hit_rate'].mean()],
        'mrr': [complete_df['mrr'].mean()],
        'ndcgs': [complete_df['ndcgs'].mean()],
        'fail': [1 - complete_df['fail'].mean()],
    })
    mean_row.to_csv(results_csv, mode='a', header=False, index=False)
    print(f'Results saved to {results_csv}')


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.query_number is not None:
        config['QUERY_NUMBER'] = args.query_number

    domain, dataset_path = resolve_dataset(config, args.data_dir)
    print('Configuration:', config)
    print('Dataset path:', dataset_path)
    print('Execution mode:', args.execution_mode)

    query_file = dataset_path / args.query_file
    df = pd.read_csv(query_file, encoding='ISO-8859-1').head(config['QUERY_NUMBER'])
    if args.classification_only or 'classification' in df.columns:
        df = df[df['classification'] == 1]

    pipeline_steps = [step.strip() for step in args.pipeline.split(',') if step.strip()]
    process_queries(df, domain, dataset_path, config, args.execution_mode, pipeline_steps)


if __name__ == '__main__':
    main()
