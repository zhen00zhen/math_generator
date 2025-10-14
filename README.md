# 小学四则运算题目生成器
一个用于生成小学四则运算题目的命令行程序，支持题目生成、答案计算和答案检查功能。

## 功能特点
- ✅ 生成符合小学要求的四则运算题目
- ✅ 支持自然数、真分数和带分数
- ✅ 严格限制运算符数量不超过3个
- ✅ 确保减法结果非负、除法结果为真分数
- ✅ 自动去重，避免重复题目
- ✅ 支持题目和答案的批量生成
- ✅ 提供答案检查功能
- ✅ 支持大量题目生成（可达10000道）

## 安装要求

### 系统要求
- Python 3.6 或更高版本
- 支持的操作系统：Windows、macOS、Linux

### 依赖安装
```bash
# 克隆项目
git clone <repository-url>
cd math-generator

# 安装依赖（项目使用纯Python标准库，无需额外安装）
# 如果使用虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

快速开始
1. 生成题目
# 生成10道10以内的题目
python math_generator.py -n 10 -r 10

# 生成50道20以内的题目
python math_generator.py -n 50 -r 20

2. 检查答案
# 检查答案
python math_generator.py -e Exercises.txt -a Answers.txt

详细使用方法
基本命令格式
python math_generator.py [选项]

生成模式
使用 -n 和 -r 参数来生成题目：
# 生成10道题目，数值范围在10以内
python math_generator.py -n 10 -r 10

# 生成100道题目，数值范围在20以内
python math_generator.py -n 100 -r 20

参数说明：
-n: 生成题目的数量（必须大于0）
-r: 数值范围（自然数、真分数和真分数分母的范围，必须大于1）

输出文件：
Exercises.txt: 生成的题目文件
Answers.txt: 对应的答案文件

检查模式
使用 -e 和 -a 参数来检查答案：
# 检查标准文件
python math_generator.py -e Exercises.txt -a Answers.txt

# 检查自定义文件
python math_generator.py -e my_exercises.txt -a my_answers.txt
参数说明：
-e: 题目文件路径
-a: 答案文件路径

输出文件：
Grade.txt: 检查结果统计

题目规范
支持的数值类型
自然数: 0, 1, 2, 3, ...
真分数: 1/2, 1/3, 2/3, 1/4, ...
带分数: 1'1/2, 2'3/4, ...

支持的运算符
加法: +
减法: -
乘法: ×
除法: ÷

题目要求
1.每道题目中运算符个数不超过3个

2.减法运算确保结果非负

3.除法运算确保结果为真分数

4.题目之间不会重复（包括交换律等数学等价形式）

项目结构
math-generator/
├── math_generator.py    # 主程序文件
├── test.py             # 测试代码
├── README.md           # 项目说明
└── requirements.txt    # 依赖说明

运行测试
bash
# 运行完整的测试套件
python test.py

# 运行特定测试类别
python test.py TestFraction
python test.py TestProblemGenerator
