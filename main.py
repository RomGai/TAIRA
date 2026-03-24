import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from agents.interact_agent import InteractorAgent
from agents.item_retrieval_agent import ItemRetrievalAgent
from agents.searcher_agent import SearcherAgent
from agents.task_interpreter_agent import InterpreterAgent
from core.manager_core import TAIRAManager
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
    parser.add_argument(
        '--final-recall-size',
        type=int,
        default=40,
        help='Maximum number of unique recalled items used for final ranking/evaluation.'
    )
    return parser.parse_args()


def _safe_item_id(value):
    if isinstance(value, dict):
        return str(value.get('id', '') or value.get('item_id', '')).strip()
    return str(value or '').strip()


def _extract_ranked_ids_from_response(final_json):
    ranked_ids = []
    for recommendation in final_json.get('recommendations', []):
        for item in recommendation.get('items', []):
            item_id = _safe_item_id(item)
            if item_id:
                ranked_ids.append(item_id)
    return ranked_ids


def _dedupe_keep_order(item_ids):
    deduped = []
    seen = set()
    for item_id in item_ids:
        if not item_id or item_id in seen:
            continue
        deduped.append(item_id)
        seen.add(item_id)
    return deduped


def _mrr_at_k(labels, k):
    for rank, label in enumerate(labels[:k], start=1):
        if int(label) == 1:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(labels, k):
    ranked = labels[:k]
    dcg = 0.0
    for rank, rel in enumerate(ranked, start=1):
        dcg += (2 ** int(rel) - 1) / math.log2(rank + 1)

    ideal = sorted((int(x) for x in labels), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal, start=1):
        idcg += (2 ** rel - 1) / math.log2(rank + 1)

    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _compute_topk_metrics(ranked_ids, target_id, top_ks=(10, 20, 40)):
    labels = [1 if item_id == target_id else 0 for item_id in ranked_ids]
    metrics = {}
    for k in top_ks:
        metrics[f'hit@{k}'] = float(sum(labels[:k]) > 0)
        metrics[f'ndcg@{k}'] = _ndcg_at_k(labels, k)
        metrics[f'mrr@{k}'] = _mrr_at_k(labels, k)
    return metrics


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


def run_pipeline_mode(memory, row, domain, config, logger, agents, pipeline_steps, final_recall_size):
    item_agent, searcher_agent, interactor_agent, _ = agents

    user_input = row['new_query']
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
                for item in retrieval_records[:final_recall_size]
            ],
        }]
        final_json = {'recommendations': recommendations}
    else:
        import re
        match = re.search(r'\{.*\}', final_response, re.DOTALL)
        if not match:
            raise ValueError('InteractorAgent did not return valid JSON content.')
        final_json = json.loads(match.group(0))

    retrieval_records = outputs.get('retrieve', [])
    retrieval_ranked_ids = [str(item['product_id']) for item in retrieval_records]
    retrieval_title_map = {str(item['product_id']): str(item.get('project_info', '')) for item in retrieval_records}

    raw_interactor_ranked_ids = _extract_ranked_ids_from_response(final_json)
    retrieval_id_set = set(retrieval_ranked_ids)
    interactor_ranked_ids = [item_id for item_id in raw_interactor_ranked_ids if item_id in retrieval_id_set]
    merged_ranked_ids = _dedupe_keep_order(interactor_ranked_ids + retrieval_ranked_ids)[:final_recall_size]

    dropped_ids = len(raw_interactor_ranked_ids) - len(interactor_ranked_ids)
    if dropped_ids > 0:
        logger.debug('Dropped %s interactor ids not found in retrieval results.', dropped_ids)

    if outputs.get('interact') or outputs.get('retrieve'):
        final_json['recommendations'] = [{
            'recommendation': 'merged pipeline ranking',
            'items': [
                {'id': item_id, 'title': retrieval_title_map.get(item_id, '')}
                for item_id in merged_ranked_ids
            ],
        }]

    target_id = str(row['id'])
    metrics = _compute_topk_metrics(merged_ranked_ids, target_id)
    fail_flag = len(merged_ranked_ids) == 0
    return metrics, fail_flag, 'direct_pipeline'


def _print_running_average(df_subset, top_ks=(10, 20, 40)):
    parts = []
    for k in top_ks:
        parts.append(
            f"Top{k} Hit={df_subset[f'hit@{k}'].mean():.4f} "
            f"NDCG={df_subset[f'ndcg@{k}'].mean():.4f} "
            f"MRR={df_subset[f'mrr@{k}'].mean():.4f}"
        )
    print('[RunningAvg] ' + ' | '.join(parts))


def process_queries(df, domain, dataset_path, config, execution_mode, pipeline_steps, final_recall_size):
    method = config['METHOD'] if execution_mode == 'manager' else f"pipeline-{'-'.join(pipeline_steps)}"
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H_%M_%S')
    log_dir = dataset_path / 'logs' / f'{method}-{formatted_time}'
    log_dir.mkdir(parents=True, exist_ok=True)
    results_csv = log_dir / f'result-{method}-{formatted_time}.csv'

    memory = Memory()
    agents = init_agents(memory)

    metric_columns = [
        'hit@10', 'ndcg@10', 'mrr@10',
        'hit@20', 'ndcg@20', 'mrr@20',
        'hit@40', 'ndcg@40', 'mrr@40',
    ]

    for index, row in df.iterrows():
        log_file = log_dir / f'log_{index + 1}.log'
        logger = setup_logger(str(log_file))
        print(f'Processing query {index + 1}')

        try:
            if execution_mode == 'manager':
                hit_rate, mrr, ndcg, fail_flag, pattern_key = run_manager_mode(
                    memory, row, domain, config, logger, agents
                )
                metrics = {
                    'hit@10': hit_rate,
                    'ndcg@10': ndcg,
                    'mrr@10': mrr,
                    'hit@20': hit_rate,
                    'ndcg@20': ndcg,
                    'mrr@20': mrr,
                    'hit@40': hit_rate,
                    'ndcg@40': ndcg,
                    'mrr@40': mrr,
                }
            else:
                metrics, fail_flag, pattern_key = run_pipeline_mode(
                    memory, row, domain, config, logger, agents, pipeline_steps, final_recall_size
                )

            for metric_key, metric_value in metrics.items():
                row[metric_key] = metric_value
            row['fail'] = 1 if fail_flag else 0
            row['pattern_used'] = pattern_key
        except Exception as exc:
            error_msg = f'Error processing query {index + 1}: {exc}'
            print(error_msg)
            logger.error(error_msg)
            for metric_key in metric_columns:
                row[metric_key] = 0
            row['fail'] = 1
            row['pattern_used'] = 'error'

        row_df = pd.DataFrame([row])
        if not results_csv.exists():
            row_df.to_csv(results_csv, mode='w', header=True, index=False)
        else:
            row_df.to_csv(results_csv, mode='a', header=False, index=False)

        complete_df = pd.read_csv(results_csv, encoding='ISO-8859-1')
        _print_running_average(complete_df)

        memory.remove_data()
        logger.handlers.clear()

    complete_df = pd.read_csv(results_csv, encoding='ISO-8859-1')
    mean_row = pd.DataFrame({
        'hit@10': [complete_df['hit@10'].mean()],
        'ndcg@10': [complete_df['ndcg@10'].mean()],
        'mrr@10': [complete_df['mrr@10'].mean()],
        'hit@20': [complete_df['hit@20'].mean()],
        'ndcg@20': [complete_df['ndcg@20'].mean()],
        'mrr@20': [complete_df['mrr@20'].mean()],
        'hit@40': [complete_df['hit@40'].mean()],
        'ndcg@40': [complete_df['ndcg@40'].mean()],
        'mrr@40': [complete_df['mrr@40'].mean()],
        'fail': [1 - complete_df['fail'].mean()],
    })
    mean_row.to_csv(results_csv, mode='a', header=False, index=False)
    print(f'Results saved to {results_csv}')


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.query_number is not None:
        config['QUERY_NUMBER'] = args.query_number

    if args.execution_mode == 'pipeline':
        config['TOPK_ITEMS'] = max(int(config.get('TOPK_ITEMS', 10)), int(args.final_recall_size))

    domain, dataset_path = resolve_dataset(config, args.data_dir)
    print('Configuration:', config)
    print('Dataset path:', dataset_path)
    print('Execution mode:', args.execution_mode)

    query_file = dataset_path / args.query_file
    df = pd.read_csv(query_file, encoding='ISO-8859-1').head(config['QUERY_NUMBER'])
    if args.classification_only or 'classification' in df.columns:
        df = df[df['classification'] == 1]

    pipeline_steps = [step.strip() for step in args.pipeline.split(',') if step.strip()]
    process_queries(df, domain, dataset_path, config, args.execution_mode, pipeline_steps, args.final_recall_size)


if __name__ == '__main__':
    main()
