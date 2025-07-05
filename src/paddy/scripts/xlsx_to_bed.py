#! /usr/bin/env python3
import pandas as pd
import argparse
import os
from pathlib import Path

class XlsxToBedConverter:
    """XLSX到BED文件转换器"""
    
    def __init__(self, xlsx_file, output_bed=None):
        self.xlsx_file = xlsx_file
        self.output_bed = output_bed or self._generate_output_name()
        self.df = None
    
    def _generate_output_name(self):
        """生成输出文件名"""
        input_path = Path(self.xlsx_file)
        return input_path.with_suffix('.bed')
    
    def load_xlsx(self, sheet_name=0, header=0):
        """加载xlsx文件"""
        print(f"读取xlsx文件: {self.xlsx_file}")
        
        try:
            self.df = pd.read_excel(self.xlsx_file, sheet_name=sheet_name, header=header)
            print(f"成功加载文件")
            print(f"数据形状: {self.df.shape}")
            print(f"列名: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"读取文件失败: {e}")
            return False
    
    def preview_data(self, rows=5):
        """预览数据"""
        if self.df is None:
            print("请先加载数据")
            return
        
        print(f"\n 数据预览（前{rows}行）:")
        print(self.df.head(rows))
        
        print(f"\n数据类型:")
        print(self.df.dtypes)
        
        print(f"\n数据统计:")
        print(self.df.describe())
    
    def extract_bed_data(self, chrom_col=4, start_col=5, end_col=6, name_col=None, score_col=None, strand_col=None, flag_col=None):
        """
        提取BED格式数据
        
        Args:
            chrom_col (int): 染色体列索引（从0开始）
            start_col (int): 起始位置列索引
            end_col (int): 结束位置列索引
            name_col (int): 名称列索引（可选）
            score_col (int): 分数列索引（可选）
            strand_col (int): 链方向列索引（可选）
            flag_col (int): Flag列索引（可选，将作为第4列）
        """
        if self.df is None:
            print("请先加载数据")
            return None
        
        print(f"\n提取列: {chrom_col+1}, {start_col+1}, {end_col+1}")
        if flag_col is not None:
            print(f"   Flag列: {flag_col+1}")
        if name_col is not None:
            print(f"   名称列: {name_col+1}")
        if score_col is not None:
            print(f"   分数列: {score_col+1}")
        if strand_col is not None:
            print(f"   链方向列: {strand_col+1}")
        
        bed_data = []
        skipped_rows = []
        
        for i, row in self.df.iterrows():
            try:
                # 提取基本列
                chrom = str(row.iloc[chrom_col])
                start = int(row.iloc[start_col])
                end = int(row.iloc[end_col])
                
                # 检查数据有效性
                if pd.isna(chrom) or pd.isna(start) or pd.isna(end):
                    skipped_rows.append((i+1, "包含空值"))
                    continue
                
                if start >= end:
                    skipped_rows.append((i+1, "起始位置 >= 结束位置"))
                    continue
                
                # 构建BED行
                bed_row = [chrom, start, end]
                
                # 添加Flag列作为第4列
                if flag_col is not None:
                    flag = str(row.iloc[flag_col]) if not pd.isna(row.iloc[flag_col]) else "."
                    bed_row.append(flag)
                else:
                    bed_row.append(".")
                
                # 添加可选列
                if name_col is not None:
                    name = str(row.iloc[name_col]) if not pd.isna(row.iloc[name_col]) else "."
                    bed_row.append(name)
                else:
                    bed_row.append(".")
                
                if score_col is not None:
                    score = int(row.iloc[score_col]) if not pd.isna(row.iloc[score_col]) else 0
                    bed_row.append(score)
                else:
                    bed_row.append(0)
                
                if strand_col is not None:
                    strand = str(row.iloc[strand_col]) if not pd.isna(row.iloc[strand_col]) else "."
                    bed_row.append(strand)
                else:
                    bed_row.append(".")
                
                bed_data.append(bed_row)
                
            except Exception as e:
                skipped_rows.append((i+1, f"处理错误: {e}"))
        
        print(f"成功提取: {len(bed_data)} 行")
        if skipped_rows:
            print(f"跳过: {len(skipped_rows)} 行")
            for row_num, reason in skipped_rows[:10]:  # 只显示前10个
                print(f"   第{row_num}行: {reason}")
            if len(skipped_rows) > 10:
                print(f"   ... 还有 {len(skipped_rows)-10} 行被跳过")
        
        return bed_data
    
    def save_bed(self, bed_data, bed_format="standard"):
        """
        保存BED文件
        
        Args:
            bed_data: BED格式数据
            bed_format: BED格式类型 ("standard", "bed6", "bed12")
        """
        if not bed_data:
            print("没有数据可保存")
            return
        
        print(f"\n保存BED文件: {self.output_bed}")
        print(f"格式: {bed_format}")
        
        try:
            with open(self.output_bed, 'w') as f:
                for row in bed_data:
                    if bed_format == "standard":
                        # 标准BED格式 (chrom, start, end)
                        f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
                    elif bed_format == "bed4":
                        # BED4格式 (chrom, start, end, flag)
                        if len(row) >= 4:
                            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")
                        else:
                            print(f"BED4格式需要4列，当前只有{len(row)}列")
                            break
                    elif bed_format == "bed6":
                        # BED6格式 (chrom, start, end, name, score, strand)
                        # 如果包含flag列，则输出为: chrom, start, end, flag, name, score
                        if len(row) >= 6:
                            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}\n")
                        else:
                            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}\n")
                    elif bed_format == "bed12":
                        # BED12格式 (需要更多列)
                        if len(row) >= 12:
                            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}\t{row[6]}\t{row[7]}\t{row[8]}\t{row[9]}\t{row[10]}\t{row[11]}\n")
                        else:
                            print(f"BED12格式需要12列，当前只有{len(row)}列")
                            break
            
            print(f"成功保存BED文件")
            print(f"文件大小: {os.path.getsize(self.output_bed) / 1024:.2f} KB")
            
            # 显示文件预览
            print(f"\n文件预览:")
            with open(self.output_bed, 'r') as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"  {line.strip()}")
                    else:
                        print(f"  ... 还有更多行")
                        break
                        
        except Exception as e:
            print(f"保存文件失败: {e}")
    
    def convert(self, chrom_col=4, start_col=5, end_col=6, name_col=None, score_col=None, strand_col=None, flag_col=None, bed_format="standard"):
        """完整的转换流程"""
        # 1. 加载数据
        if not self.load_xlsx():
            return False
        
        # 2. 预览数据
        self.preview_data()
        
        # 3. 提取BED数据
        bed_data = self.extract_bed_data(chrom_col, start_col, end_col, name_col, score_col, strand_col, flag_col)
        
        if bed_data is None:
            return False
        
        # 4. 保存BED文件
        self.save_bed(bed_data, bed_format)
        
        return True

# 命令行版本
def main():
    parser = argparse.ArgumentParser(description="将xlsx文件转换为BED格式")
    parser.add_argument("xlsx_file", help="输入的xlsx文件路径")
    parser.add_argument("-o", "--output", help="输出的BED文件路径")
    parser.add_argument("--chrom", type=int, default=4, help="染色体列索引（从0开始，默认4）")
    parser.add_argument("--start", type=int, default=5, help="起始位置列索引（从0开始，默认5）")
    parser.add_argument("--end", type=int, default=6, help="结束位置列索引（从0开始，默认6）")
    parser.add_argument("--name", type=int, help="名称列索引（可选）")
    parser.add_argument("--score", type=int, help="分数列索引（可选）")
    parser.add_argument("--strand", type=int, help="链方向列索引（可选）")
    parser.add_argument("--flag", type=int, help="Flag列索引（可选，将作为bed的第4列），如果flag列存在，则--format建议为bed4")
    parser.add_argument("--format", choices=["standard", "bed6", "bed12", "bed4"], default="standard", help="BED格式类型")
    parser.add_argument("--sheet", type=int, default=0, help="工作表索引（默认0）")
    parser.add_argument("--no-header", action="store_true", help="不包含表头")
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = XlsxToBedConverter(args.xlsx_file, args.output)
    
    # 加载数据
    if not converter.load_xlsx(sheet_name=args.sheet, header=None if args.no_header else 0):
        return
    
    # 执行转换
    converter.convert(
        chrom_col=args.chrom,
        start_col=args.start,
        end_col=args.end,
        name_col=args.name,
        score_col=args.score,
        strand_col=args.strand,
        flag_col=args.flag,
        bed_format=args.format
    )

# 使用示例
if __name__ == "__main__":
    main()
    # 方法1: 直接使用
    # xlsx_file = "your_file.xlsx"
    # converter = XlsxToBedConverter(xlsx_file)
    # converter.convert(chrom_col=4, start_col=5, end_col=6, flag_col=7, bed_format="bed4")
    
    # 方法2: 命令行使用
    # python xlsx_to_bed.py your_file.xlsx -o output.bed --chrom 4 --start 5 --end 6 --flag 7 --format bed4