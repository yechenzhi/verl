import argparse
import os

import datasets
from datasets import Dataset

def main():
    """
    主函数，用于读取、去重并保存AIME 2024 Parquet数据集。
    """
    parser = argparse.ArgumentParser(
        description="读取一个Parquet文件，根据 'extra_info.raw_problem' 字段去重，并保存为新文件。"
    )
    # 接收输入文件路径作为参数
    parser.add_argument(
        "--input_file",
        required=True,
        help="需要去重的源Parquet文件路径 (例如: /root/highspeedstorage/h800/data/aime-2024.parquet)",
    )
    # 接收输出文件路径作为参数
    parser.add_argument(
        "--output_file",
        required=True,
        help="去重后保存的新Parquet文件路径 (例如: /root/highspeedstorage/h800/data/aime-2024-deduplicated.parquet)",
    )
    args = parser.parse_args()

    print(f"正在从 '{args.input_file}' 加载数据集...")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件未找到 at '{args.input_file}'")
        return

    # 使用datasets库加载Parquet文件
    original_dataset = datasets.load_dataset("parquet", data_files=args.input_file)["train"]
    
    print(f"加载成功。去重前的数据集行数: {len(original_dataset)}")
    print("数据集结构预览:")
    print(original_dataset)

    # --- 去重逻辑 ---
    print("\n开始去重...")
    
    # 使用一个set来高效地追踪已经见过的 'raw_problem'
    seen_problems = set()
    
    # 定义一个过滤函数。datasets的 .filter() 方法会遍历每一行，
    # 并根据函数的返回值 (True/False) 决定是否保留该行。
    def is_unique(example):
        """
        检查一个数据条目是否是唯一的。

        Args:
            example (dict): 数据集中的一行。

        Returns:
            bool: 如果 'raw_problem' 是第一次出现，则返回True，否则返回False。
        """
        # 我们使用 extra_info 字典中的 'raw_problem' 作为唯一标识符
        problem_text = example['extra_info']['raw_problem']
        
        if problem_text in seen_problems:
            # 如果已经见过这个问题，返回False，此行将被丢弃
            return False
        else:
            # 如果是新问题，将其添加到set中，并返回True，此行将被保留
            seen_problems.add(problem_text)
            return True

    # 应用过滤器，这会返回一个新的、只包含唯一行的数据集
    deduplicated_dataset = original_dataset.filter(is_unique)

    print("去重完成。")
    print(f"去重后的数据集行数: {len(deduplicated_dataset)}")

    # --- 保存结果 ---
    # 确保输出文件的目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n正在将去重后的数据集保存到: '{args.output_file}'")
    
    # 将干净的数据集保存到新的Parquet文件
    deduplicated_dataset.to_parquet(args.output_file)
    
    print("文件保存成功！脚本执行完毕。")


if __name__ == "__main__":
    main()