import argparse
import os

import datasets

def main():
    """
    主函数，用于加载、转换和保存AIME 2025数据集。
    """
    parser = argparse.ArgumentParser(
        description="将Hugging Face上的AIME 2025数据集转换为与AIME 2024格式兼容的Parquet文件。"
    )
    parser.add_argument(
        "--output_dir",
        default="data/aime2025_processed",
        help="保存输出Parquet文件的本地目录。",
    )
    parser.add_argument(
        "--output_filename",
        default="aime-2025.parquet",
        help="输出Parquet文件的名称。",
    )
    args = parser.parse_args()

    source_dataset_name = "yentinglin/aime_2025"
    target_data_source = "AIME2025"

    print(f"正在从Hugging Face Hub加载数据集: {source_dataset_name}...")
    dataset = datasets.load_dataset(source_dataset_name, "default")
    print("数据集加载成功。")

    prompt_prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
    prompt_suffix = "Remember to put your answer on its own line after \"Answer:\"."

    def format_aime_entry(example, idx):
        """
        将AIME 2025数据集中的单个条目转换为目标格式。
        """
        raw_problem = example.pop("problem")
        prompt_content = f"{prompt_prefix}\n\n{raw_problem}\n\n{prompt_suffix}"
        ground_truth_answer = example.pop("answer")

        # --- 核心改动点 ---
        # 我们严格按照报错信息中 "output fields" 的顺序来定义 extra_info 字典的键。
        # 目标顺序: index, raw_problem, split, original_id, url, year
        formatted_data = {
            "data_source": target_data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ],
            "ability": "MATH",
            "reward_model": {
                "style": "rule-lighteval/MATH_v2",
                "ground_truth": ground_truth_answer,
            },
            "extra_info": {
                "index": idx,
                "raw_problem": raw_problem,
                "split": "train",
                "url": example.get("url"),
                "year": example.get("year"),
            },
        }
        return formatted_data

    train_dataset = dataset["train"]
    print("正在转换数据集格式...")
    processed_dataset = train_dataset.map(
        function=format_aime_entry,
        with_indices=True,
        remove_columns=train_dataset.column_names
    )
    print("格式转换完成。")
    print("转换后的数据集结构:")
    print(processed_dataset)

    output_path = os.path.join(args.output_dir, args.output_filename)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n正在将转换后的数据集保存到: {output_path}")
    processed_dataset.to_parquet(output_path)
    print("文件保存成功！脚本执行完毕。")


if __name__ == "__main__":
    main()