import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from main import (
    _compute_topk_metrics,
    _dedupe_keep_order,
    _extract_ranked_ids_from_response,
    _print_running_average,
    init_agents,
    load_config,
    resolve_dataset,
    setup_logger,
)
from utils.memory import Memory


pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.unicode.east_asian_width', True)


def parse_args():
    parser = argparse.ArgumentParser(description='Run TAIRA in ReAct-style iterative execution.')
    parser.add_argument('--config', default='system_config.yaml', help='Path to the YAML config file.')
    parser.add_argument('--data-dir', help='Dataset directory under data/, e.g. data/amazon_music.')
    parser.add_argument('--query-file', default='query_data1.csv', help='Query CSV filename inside the dataset directory.')
    parser.add_argument('--pipeline', default='search,retrieve,interact', help='Pipeline actions to allow.')
    parser.add_argument('--query-number', type=int, help='Optional override for QUERY_NUMBER.')
    parser.add_argument('--classification-only', action='store_true', help='Only run rows where classification == 1.')
    parser.add_argument('--final-recall-size', type=int, default=40, help='Maximum number of unique recalled items.')
    parser.add_argument('--agent-recall-size', type=int, default=10, help='Number of final recalled items from interactor output.')
    parser.add_argument('--agent-use-all-recommendation-groups', action='store_true', help='Use all recommendation groups.')
    parser.add_argument('--max-turns', type=int, default=3, help='Maximum ReAct turns for one query.')
    parser.add_argument('--use-openai-gemini', action='store_true', help='Route Qwen/Qwen3-8B inference to Gemini endpoint.')
    return parser.parse_args()


def _parse_interactor_json(raw_response):
    import re

    match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    if not match:
        return {'recommendations': []}
    return json.loads(match.group(0))


def run_react_query(memory, row, agents, logger, pipeline_steps, max_turns, final_recall_size, agent_recall_size, use_all_agent_groups):
    item_agent, searcher_agent, interactor_agent, _ = agents
    user_input = row['new_query']
    target_id = str(row['id'])

    memory.add_input(user_input)
    current_query = user_input
    retrieval_records = []
    final_json = {'recommendations': []}

    for turn in range(1, max_turns + 1):
        if 'search' in pipeline_steps and (turn == 1 or not retrieval_records):
            thought = f'Turn {turn} thought: need more external knowledge.'
            memory.add_thought({'react_thought': thought})
            logger.debug(thought)
            current_query = str(searcher_agent.execute_task(current_query))
            memory.add_observation('SearcherAgent', 'search', current_query)
            logger.debug('Turn %s action=search observation=%s', turn, current_query)

        if 'retrieve' in pipeline_steps:
            thought = f'Turn {turn} thought: retrieve candidate items from current hypothesis.'
            memory.add_thought({'react_thought': thought})
            logger.debug(thought)
            retrieval_df = item_agent.execute_task(current_query)
            retrieval_records = retrieval_df.to_dict(orient='records')
            memory.add_observation('ItemRetrievalAgent', current_query, retrieval_records)
            logger.debug('Turn %s action=retrieve candidates=%s', turn, len(retrieval_records))

        if 'interact' in pipeline_steps:
            thought = f'Turn {turn} thought: aggregate observations and produce recommendation.'
            memory.add_thought({'react_thought': thought})
            logger.debug(thought)
            instruction = f'ReAct turn {turn}: produce recommendation for query: {user_input}'
            final_response = interactor_agent.generate_response(instruction)
            memory.add_observation('InteractorAgent', instruction, final_response)
            final_json = _parse_interactor_json(final_response)

            raw_ids = _extract_ranked_ids_from_response(
                final_json,
                max_groups=None if use_all_agent_groups else 1,
            )
            if raw_ids:
                logger.debug('Turn %s reached non-empty final recommendation, stop iteration.', turn)
                break

    retrieval_ranked_ids = [str(item['product_id']) for item in retrieval_records]
    retrieval_title_map = {str(item['product_id']): str(item.get('project_info', '')) for item in retrieval_records}

    raw_interactor_ranked_ids = _extract_ranked_ids_from_response(
        final_json,
        max_groups=None if use_all_agent_groups else 1,
    )
    retrieval_id_set = set(retrieval_ranked_ids)
    interactor_ranked_ids = [item_id for item_id in raw_interactor_ranked_ids if item_id in retrieval_id_set]
    selected_agent_ids = _dedupe_keep_order(interactor_ranked_ids)[:agent_recall_size]
    selected_retrieval_ids = [item_id for item_id in retrieval_ranked_ids if item_id not in set(selected_agent_ids)]
    merged_ranked_ids = _dedupe_keep_order(selected_agent_ids + selected_retrieval_ids)[:final_recall_size]

    final_json['recommendations'] = [{
        'recommendation': 'react merged ranking',
        'items': [{'id': item_id, 'title': retrieval_title_map.get(item_id, '')} for item_id in merged_ranked_ids],
    }]

    metrics = _compute_topk_metrics(merged_ranked_ids, target_id)
    fail_flag = len(merged_ranked_ids) == 0
    return metrics, fail_flag, 'react_pipeline'


def process_queries(df, dataset_path, config, args):
    method = f"react-{'-'.join([s.strip() for s in args.pipeline.split(',') if s.strip()])}"
    formatted_time = datetime.now().strftime('%Y-%m-%d %H_%M_%S')
    log_dir = dataset_path / 'logs' / f'{method}-{formatted_time}'
    log_dir.mkdir(parents=True, exist_ok=True)
    results_csv = log_dir / f'result-{method}-{formatted_time}.csv'

    memory = Memory()
    agents = init_agents(memory, config)
    metric_columns = ['hit@10', 'ndcg@10', 'mrr@10', 'hit@20', 'ndcg@20', 'mrr@20', 'hit@40', 'ndcg@40', 'mrr@40']
    pipeline_steps = [step.strip() for step in args.pipeline.split(',') if step.strip()]

    for index, row in df.iterrows():
        logger = setup_logger(str(log_dir / f'log_{index + 1}.log'))
        print(f'Processing query {index + 1}')
        try:
            metrics, fail_flag, pattern_key = run_react_query(
                memory,
                row,
                agents,
                logger,
                pipeline_steps,
                args.max_turns,
                args.final_recall_size,
                args.agent_recall_size,
                args.agent_use_all_recommendation_groups,
            )
            for k, v in metrics.items():
                row[k] = v
            row['fail'] = 1 if fail_flag else 0
            row['pattern_used'] = pattern_key
        except Exception as exc:
            logger.error('Error processing query %s: %s', index + 1, exc)
            for metric_key in metric_columns:
                row[metric_key] = 0
            row['fail'] = 1
            row['pattern_used'] = 'error'

        row_df = pd.DataFrame([row])
        row_df.to_csv(results_csv, mode='w' if not results_csv.exists() else 'a', header=not results_csv.exists(), index=False)

        complete_df = pd.read_csv(results_csv, encoding='ISO-8859-1')
        _print_running_average(complete_df)
        memory.remove_data()
        logger.handlers.clear()

    complete_df = pd.read_csv(results_csv, encoding='ISO-8859-1')
    mean_row = pd.DataFrame({
        'hit@10': [complete_df['hit@10'].mean()], 'ndcg@10': [complete_df['ndcg@10'].mean()], 'mrr@10': [complete_df['mrr@10'].mean()],
        'hit@20': [complete_df['hit@20'].mean()], 'ndcg@20': [complete_df['ndcg@20'].mean()], 'mrr@20': [complete_df['mrr@20'].mean()],
        'hit@40': [complete_df['hit@40'].mean()], 'ndcg@40': [complete_df['ndcg@40'].mean()], 'mrr@40': [complete_df['mrr@40'].mean()],
        'fail': [1 - complete_df['fail'].mean()],
    })
    mean_row.to_csv(results_csv, mode='a', header=False, index=False)
    print(f'Results saved to {results_csv}')


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.query_number is not None:
        config['QUERY_NUMBER'] = args.query_number
    if args.use_openai_gemini:
        os.environ['TAIRA_USE_OPENAI_GEMINI'] = '1'
        config['USE_OPENAI_GEMINI'] = True

    if args.agent_recall_size < 0:
        raise ValueError('--agent-recall-size must be >= 0.')
    if args.agent_recall_size > args.final_recall_size:
        raise ValueError('--agent-recall-size cannot be larger than --final-recall-size.')
    config['TOPK_ITEMS'] = max(int(config.get('TOPK_ITEMS', 10)), int(args.final_recall_size))

    _, dataset_path = resolve_dataset(config, args.data_dir)
    df = pd.read_csv(dataset_path / args.query_file, encoding='ISO-8859-1').head(config['QUERY_NUMBER'])
    if args.classification_only or 'classification' in df.columns:
        df = df[df['classification'] == 1]

    process_queries(df, dataset_path, config, args)


if __name__ == '__main__':
    main()
