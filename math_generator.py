
import random
import math
import os
from functools import total_ordering
import sys
import locale

@total_ordering
class Fraction:
    """分数类，处理真分数和带分数"""
    
    def __init__(self, numerator, denominator=1):
        if denominator == 0:
            raise ValueError("分母不能为零")
        
        self.numerator = numerator
        self.denominator = denominator
        self._simplify()
    
    def _simplify(self):
        """约分"""
        if self.numerator == 0:
            self.denominator = 1
            return
            
        gcd = math.gcd(abs(self.numerator), self.denominator)
        if gcd > 0:
            self.numerator //= gcd
            self.denominator //= gcd
        
        # 确保分母为正
        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator
    
    @classmethod
    def from_string(cls, s):
        """从字符串创建分数"""
        s = str(s).strip()
        if "'" in s:
            # 带分数格式: a'b/c
            integer_part, fraction_part = s.split("'")
            numerator, denominator = map(int, fraction_part.split('/'))
            integer_val = int(integer_part)
            if integer_val < 0:
                return cls(integer_val * denominator - numerator, denominator)
            else:
                return cls(integer_val * denominator + numerator, denominator)
        elif '/' in s:
            # 真分数格式: a/b
            numerator, denominator = map(int, s.split('/'))
            return cls(numerator, denominator)
        else:
            # 整数
            return cls(int(s))
    
    def to_string(self):
        """转换为字符串表示"""
        if self.numerator == 0:
            return "0"
            
        if self.denominator == 1:
            return str(self.numerator)
        
        abs_numerator = abs(self.numerator)
        if abs_numerator >= self.denominator:
            # 带分数
            integer_part = self.numerator // self.denominator
            remainder = abs_numerator % self.denominator
            if remainder == 0:
                return str(integer_part)
            else:
                return f"{integer_part}'{remainder}/{self.denominator}"
        else:
            # 真分数
            return f"{self.numerator}/{self.denominator}"
    
    def is_proper_fraction(self):
        """检查是否为真分数（分子绝对值小于分母）"""
        return abs(self.numerator) < self.denominator
    
    def is_non_negative(self):
        """检查是否非负"""
        return self.numerator >= 0
    
    def __add__(self, other):
        other = self._ensure_fraction(other)
        new_numerator = (self.numerator * other.denominator + 
                        other.numerator * self.denominator)
        new_denominator = self.denominator * other.denominator
        return Fraction(new_numerator, new_denominator)
    
    def __sub__(self, other):
        other = self._ensure_fraction(other)
        new_numerator = (self.numerator * other.denominator - 
                        other.numerator * self.denominator)
        new_denominator = self.denominator * other.denominator
        result = Fraction(new_numerator, new_denominator)
        if not result.is_non_negative():
            raise ValueError("减法结果不能为负数")
        return result
    
    def __mul__(self, other):
        other = self._ensure_fraction(other)
        return Fraction(self.numerator * other.numerator, 
                       self.denominator * other.denominator)
    
    def __truediv__(self, other):
        other = self._ensure_fraction(other)
        if other.numerator == 0:
            raise ValueError("除数不能为零")
        result = Fraction(self.numerator * other.denominator, 
                       self.denominator * other.numerator)
        # 移除这里的检查，因为在表达式级别已经检查
        return result
    
    def _ensure_fraction(self, value):
        """确保输入值为Fraction类型"""
        if isinstance(value, int):
            return Fraction(value)
        return value
    
    def __eq__(self, other):
        if other is None:
            return False
        other = self._ensure_fraction(other)
        return (self.numerator * other.denominator == 
                other.numerator * self.denominator)
    
    def __lt__(self, other):
        other = self._ensure_fraction(other)
        return (self.numerator * other.denominator < 
                other.numerator * self.denominator)
    
    def __str__(self):
        return self.to_string()
    
    def __repr__(self):
        return f"Fraction({self.numerator}, {self.denominator})"

class ExpressionParser:
    """表达式解析器，使用递归下降解析"""
    
    def __init__(self):
        self.tokens = []
        self.current_token = None
        self.index = 0
    
    def parse(self, expression_str):
        """解析表达式字符串"""
        # 清理表达式字符串
        expr_str = expression_str.strip()
        if '. ' in expr_str:
            expr_str = expr_str.split('. ', 1)[1]
        expr_str = expr_str.rstrip(' =')
        
        # 分词
        self.tokens = self.tokenize(expr_str)
        self.index = 0
        self.current_token = self.tokens[0] if self.tokens else None
        
        # 解析表达式
        return self.parse_expression()
    
    def tokenize(self, expr_str):
        """将表达式字符串转换为token列表"""
        tokens = []
        i = 0
        n = len(expr_str)
        
        while i < n:
            char = expr_str[i]
            
            if char.isspace():
                i += 1
                continue
                
            if char in '()':
                tokens.append(('paren', char))
                i += 1
            elif char in '+-×÷':
                tokens.append(('operator', char))
                i += 1
            else:
                # 解析数字或分数
                start = i
                while i < n and (expr_str[i].isdigit() or expr_str[i] in "'/"):
                    i += 1
                token_value = expr_str[start:i]
                
                # 检查是否是有效的数字/分数
                if self.is_number(token_value):
                    tokens.append(('number', token_value))
                else:
                    raise ValueError(f"无效的数字格式: {token_value}")
        
        return tokens
    
    def is_number(self, s):
        """检查字符串是否是有效的数字或分数"""
        s = s.strip()
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return True
        if "'" in s:
            parts = s.split("'")
            if len(parts) == 2 and parts[0].replace('-', '').isdigit():
                return '/' in parts[1] and all(p.isdigit() for p in parts[1].split('/'))
        if '/' in s:
            parts = s.split('/')
            return len(parts) == 2 and all(p.replace('-', '').isdigit() for p in parts)
        return False
    
    def advance(self):
        """移动到下一个token"""
        self.index += 1
        if self.index < len(self.tokens):
            self.current_token = self.tokens[self.index]
        else:
            self.current_token = None
    
    def parse_expression(self):
        """解析表达式"""
        return self.parse_add_sub()
    
    def parse_add_sub(self):
        """解析加减法表达式"""
        left = self.parse_mul_div()
        
        while self.current_token and self.current_token[0] == 'operator' and self.current_token[1] in '+-':
            operator = self.current_token[1]
            self.advance()
            right = self.parse_mul_div()
            left = (operator, left, right)
        
        return left
    
    def parse_mul_div(self):
        """解析乘除法表达式"""
        left = self.parse_primary()
        
        while self.current_token and self.current_token[0] == 'operator' and self.current_token[1] in '×÷':
            operator = self.current_token[1]
            self.advance()
            right = self.parse_primary()
            left = (operator, left, right)
        
        return left
    
    def parse_primary(self):
        """解析基本元素（数字或括号表达式）"""
        if self.current_token[0] == 'number':
            value = Fraction.from_string(self.current_token[1])
            self.advance()
            return value
        elif self.current_token[0] == 'paren' and self.current_token[1] == '(':
            self.advance()  # 跳过 '('
            expr = self.parse_expression()
            if self.current_token[0] != 'paren' or self.current_token[1] != ')':
                raise ValueError("缺少右括号")
            self.advance()  # 跳过 ')'
            return expr
        else:
            raise ValueError(f"意外的token: {self.current_token}")
    
    def evaluate_expression_tree(self, node):
        """计算表达式树的值（不检查条件，用于比较）"""
        if isinstance(node, Fraction):
            return node
        
        operator, left, right = node
        left_val = self.evaluate_expression_tree(left)
        right_val = self.evaluate_expression_tree(right)
        
        if operator == '+':
            return left_val + right_val
        elif operator == '-':
            return left_val - right_val
        elif operator == '×':
            return left_val * right_val
        elif operator == '÷':
            if right_val.numerator == 0:
                raise ValueError("除数不能为零")
            return left_val / right_val
    
    def evaluate_with_validation(self, node):
        """计算表达式的值并进行完整的条件检查"""
        if isinstance(node, Fraction):
            return node
        
        operator, left, right = node
        left_val = self.evaluate_with_validation(left)
        right_val = self.evaluate_with_validation(right)
        
        if operator == '+':
            return left_val + right_val
        elif operator == '-':
            # 确保左值 >= 右值
            if left_val < right_val:
                raise ValueError("减法运算中左值必须大于等于右值")
            return left_val - right_val
        elif operator == '×':
            return left_val * right_val
        elif operator == '÷':
            if right_val.numerator == 0:
                raise ValueError("除数不能为零")
            # 确保除法结果为真分数
            result = left_val / right_val
            if not result.is_proper_fraction():
                raise ValueError("除法结果必须为真分数")
            # 额外检查：确保不是0除以任何数（因为结果是0，不是真分数）
            if left_val.numerator == 0:
                raise ValueError("除法运算中左值不能为0")
            return result

class ProblemGenerator:
    """题目生成器"""
    
    def __init__(self, range_limit):
        self.range_limit = range_limit
        self.operators = ['+', '-', '×', '÷']
        self.generated_expressions = set()
        self.parser = ExpressionParser()
    
    def generate_number(self):
        """生成数字（自然数或真分数）"""
        if random.random() < 0.4:  # 40%概率生成分数
            # 分母范围：2到range_limit-1，确保分母在范围内
            denominator = random.randint(2, self.range_limit - 1)
            numerator = random.randint(1, denominator - 1)
            return Fraction(numerator, denominator)
        else:
            # 整数范围：0到range_limit-1
            return Fraction(random.randint(0, self.range_limit - 1))
    
    def compare_expression_values(self, expr1, expr2):
        """比较两个表达式树的值"""
        try:
            val1 = self.parser.evaluate_expression_tree(expr1)
            val2 = self.parser.evaluate_expression_tree(expr2)
            if val1 < val2:
                return -1
            elif val1 > val2:
                return 1
            else:
                return 0
        except:
            return 0  # 如果计算失败，认为相等
    
    def generate_valid_expression_tree(self, max_operators=3, current_operators=0):
        """递归生成有效的表达式树"""
        # 修复：确保不会生成超过3个运算符的表达式
        if current_operators >= max_operators:
            return self.generate_number()
        
        # 在接近最大运算符数量时，增加生成数字的概率
        if current_operators > 0 and random.random() < (0.4 + current_operators * 0.3):
            return self.generate_number()
        
        # 选择运算符，减少除法的概率
        weights = [0.35, 0.35, 0.25, 0.05]  # +, -, ×, ÷ 的权重，进一步降低除法概率
        operator = random.choices(self.operators, weights=weights)[0]
        
        max_attempts = 50  # 增加尝试次数
        for attempt in range(max_attempts):
            try:
                # 生成左右子树 - 修复：确保不会递归生成过多运算符
                left_operators = random.randint(0, max_operators - current_operators - 1)
                right_operators = max_operators - current_operators - 1 - left_operators
                
                left = self.generate_valid_expression_tree(max_operators, current_operators + left_operators + 1)
                right = self.generate_valid_expression_tree(max_operators, current_operators + right_operators + 1)
                
                # 对于减法和除法，需要特殊处理
                if operator == '-':
                    # 确保左值 >= 右值
                    if self.compare_expression_values(left, right) < 0:
                        left, right = right, left
                
                elif operator == '÷':
                    # 确保除法结果为真分数
                    # 对于除法，我们需要确保左值 < 右值，这样结果才是真分数
                    if self.compare_expression_values(left, right) >= 0:
                        # 交换左右值，确保左值 < 右值
                        left, right = right, left
                    
                    # 额外检查：确保右值不为0，且左值不为0（因为0除以任何数得0，不是真分数）
                    right_val = self.parser.evaluate_expression_tree(right)
                    left_val = self.parser.evaluate_expression_tree(left)
                    if right_val.numerator == 0 or left_val.numerator == 0:
                        continue
                    
                    # 检查除法结果是否为真分数
                    division_result = left_val / right_val
                    if not division_result.is_proper_fraction():
                        continue
                
                # 构建表达式并验证
                expr_tree = (operator, left, right)
                result = self.parser.evaluate_with_validation(expr_tree)
                
                # 额外检查：确保除法结果是真分数（双重检查）
                if operator == '÷' and not result.is_proper_fraction():
                    continue
                
                # 如果验证通过，返回表达式树
                return expr_tree
                
            except (ValueError, ZeroDivisionError):
                # 如果验证失败，继续尝试
                continue
        
        # 如果多次尝试都失败，返回一个简单的数字
        return self.generate_number()
    
    def tree_to_string(self, node, parent_priority=0):
        """将表达式树转换为字符串"""
        if isinstance(node, Fraction):
            return str(node)
        
        operator, left, right = node
        
        # 定义运算符优先级
        priority = {'+': 1, '-': 1, '×': 2, '÷': 2}
        
        left_str = self.tree_to_string(left, priority[operator])
        right_str = self.tree_to_string(right, priority[operator] + 0.5)
        
        # 根据优先级决定是否加括号
        current_priority = priority[operator]
        expr_str = f"{left_str} {operator} {right_str}"
        
        if current_priority < parent_priority:
            return f"({expr_str})"
        else:
            return expr_str
    
    def count_operators(self, node):
        """计算表达式中的运算符数量"""
        if isinstance(node, Fraction):
            return 0
        
        operator, left, right = node
        return 1 + self.count_operators(left) + self.count_operators(right)
    
    def normalize_expression(self, node):
        """规范化表达式用于去重"""
        if isinstance(node, Fraction):
            return str(node)
        
        operator, left, right = node
        
        left_norm = self.normalize_expression(left)
        right_norm = self.normalize_expression(right)
        
        # 对于加法和乘法，按规范顺序排序子表达式
        if operator in ['+', '×']:
            # 递归规范化子表达式
            left_parts = self.get_operand_parts(left, operator)
            right_parts = self.get_operand_parts(right, operator)
            
            all_parts = left_parts + right_parts
            sorted_parts = sorted(all_parts)
            
            # 重建规范化表达式
            if len(sorted_parts) == 1:
                return sorted_parts[0]
            else:
                result = sorted_parts[0]
                for part in sorted_parts[1:]:
                    result = f"({result}{operator}{part})"
                return result
        else:
            # 对于减法和除法，保持原顺序
            return f"({left_norm}{operator}{right_norm})"
    
    def get_operand_parts(self, node, target_operator):
        """获取表达式中指定运算符的操作数部分"""
        if isinstance(node, Fraction):
            return [self.normalize_expression(node)]
        
        operator, left, right = node
        
        if operator == target_operator:
            return self.get_operand_parts(left, target_operator) + self.get_operand_parts(right, target_operator)
        else:
            return [self.normalize_expression(node)]
    
    def generate_unique_expression(self, max_attempts=100):
        """生成唯一的表达式"""
        for _ in range(max_attempts):
            try:
                expr_tree = self.generate_valid_expression_tree()
                
                # 检查运算符数量
                op_count = self.count_operators(expr_tree)
                if op_count > 3 or op_count == 0:
                    continue
                
                # 计算表达式值
                value = self.parser.evaluate_with_validation(expr_tree)
                
                # 生成规范化字符串
                norm_str = self.normalize_expression(expr_tree)
                
                # 检查是否重复
                if norm_str not in self.generated_expressions:
                    self.generated_expressions.add(norm_str)
                    expr_str = self.tree_to_string(expr_tree)
                    # 修复：返回包含等号的完整题目格式
                    return f"{expr_str} =", value
                    
            except (ValueError, ZeroDivisionError) as e:
                continue
        
        return None, None
    
    def generate_problems(self, count):
        """生成指定数量的题目"""
        problems = []
        answers = []
        
        generated_count = 0
        attempts = 0
        max_total_attempts = count * 200  # 增加最大尝试次数
        
        while generated_count < count and attempts < max_total_attempts:
            expr, answer = self.generate_unique_expression()
            if expr is not None:
                problems.append(expr)
                answers.append(answer.to_string())
                generated_count += 1
            attempts += 1
        
        if generated_count < count:
            print(f"警告: 只生成了 {generated_count} 道题目，未能达到 {count} 道")
        
        return problems, answers

class FileHandler:
    """文件处理类"""
    
    @staticmethod
    def save_problems(problems, filename='Exercises.txt'):
        """保存题目到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for i, problem in enumerate(problems, 1):
                    # 修复：问题已经包含等号，直接写入
                    f.write(f"{i}. {problem}\n")
        except Exception as e:
            print(f"保存题目文件时出错: {e}")
            # 尝试使用系统默认编码
            try:
                encoding = locale.getpreferredencoding()
                with open(filename, 'w', encoding=encoding) as f:
                    for i, problem in enumerate(problems, 1):
                        f.write(f"{i}. {problem}\n")
            except Exception as e2:
                print(f"使用默认编码保存也失败: {e2}")
    
    @staticmethod
    def save_answers(answers, filename='Answers.txt'):
        """保存答案到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for i, answer in enumerate(answers, 1):
                    f.write(f"{i}. {answer}\n")
        except Exception as e:
            print(f"保存答案文件时出错: {e}")
            # 尝试使用系统默认编码
            try:
                encoding = locale.getpreferredencoding()
                with open(filename, 'w', encoding=encoding) as f:
                    for i, answer in enumerate(answers, 1):
                        f.write(f"{i}. {answer}\n")
            except Exception as e2:
                print(f"使用默认编码保存也失败: {e2}")
    
    @staticmethod
    def load_file(filename):
        """读取文件内容"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines()]
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试系统默认编码
            try:
                encoding = locale.getpreferredencoding()
                with open(filename, 'r', encoding=encoding) as f:
                    return [line.strip() for line in f.readlines()]
            except FileNotFoundError:
                print(f"错误: 文件 {filename} 不存在")
                return []
        except FileNotFoundError:
            print(f"错误: 文件 {filename} 不存在")
            return []

class AnswerChecker:
    """答案检查器"""
    
    def __init__(self):
        self.parser = ExpressionParser()
    
    def check_answers(self, exercise_file, answer_file):
        """检查答案"""
        exercises = FileHandler.load_file(exercise_file)
        submitted_answers = FileHandler.load_file(answer_file)
        
        if not exercises or not submitted_answers:
            return [], []
        
        correct = []
        wrong = []
        
        for i in range(min(len(exercises), len(submitted_answers))):
            exercise = exercises[i]
            submitted_line = submitted_answers[i]
            
            try:
                # 解析题目并计算正确答案
                expr_tree = self.parser.parse(exercise)
                correct_answer = self.parser.evaluate_with_validation(expr_tree)
                
                # 解析提交的答案
                if '. ' in submitted_line:
                    submitted_answer_str = submitted_line.split('. ', 1)[1]
                else:
                    submitted_answer_str = submitted_line
                
                submitted_answer = Fraction.from_string(submitted_answer_str)
                
                # 比较答案
                if correct_answer == submitted_answer:
                    correct.append(i + 1)
                else:
                    wrong.append(i + 1)
                    
            except Exception as e:
                # 如果解析或计算出错，标记为错误
                wrong.append(i + 1)
        
        return correct, wrong

class MathProblemGenerator:
    """主程序类"""
    
    def __init__(self):
        self.answer_checker = AnswerChecker()
    
    def generate_mode(self, count, range_limit):
        """生成模式"""
        print(f"正在生成 {count} 道题目，数值范围: {range_limit}")
        
        problem_generator = ProblemGenerator(range_limit)
        problems, answers = problem_generator.generate_problems(count)
        
        FileHandler.save_problems(problems, 'Exercises.txt')
        FileHandler.save_answers(answers, 'Answers.txt')
        
        print(f"生成完成！题目保存在 Exercises.txt，答案保存在 Answers.txt")
        
        # 显示前几道题目和答案作为验证
        print("\n前5道题目示例:")
        for i in range(min(5, len(problems))):
            print(f"{i+1}. {problems[i]} {answers[i]}")
    
    def check_mode(self, exercise_file, answer_file):
        """检查模式"""
        print(f"正在检查答案...")
        
        correct, wrong = self.answer_checker.check_answers(exercise_file, answer_file)
        
        # 保存统计结果
        try:
            with open('Grade.txt', 'w', encoding='utf-8') as f:
                f.write(f"Correct: {len(correct)} ({', '.join(map(str, sorted(correct)))})\n")
                f.write(f"Wrong: {len(wrong)} ({', '.join(map(str, sorted(wrong)))})\n")
        except:
            encoding = locale.getpreferredencoding()
            with open('Grade.txt', 'w', encoding=encoding) as f:
                f.write(f"Correct: {len(correct)} ({', '.join(map(str, sorted(correct)))})\n")
                f.write(f"Wrong: {len(wrong)} ({', '.join(map(str, sorted(wrong)))})\n")
        
        print(f"检查完成！结果保存在 Grade.txt")
        print(f"正确: {len(correct)}题, 错误: {len(wrong)}题")

def main():
    parser = argparse.ArgumentParser(
        description='小学四则运算题目生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s -n 10 -r 10     生成10道10以内的题目
  %(prog)s -e exercises.txt -a answers.txt  检查答案
        '''
    )
    
    # 生成模式参数
    parser.add_argument('-n', type=int, help='生成题目的数量')
    parser.add_argument('-r', type=int, help='数值范围（自然数、真分数和真分数分母的范围）')
    
    # 检查模式参数
    parser.add_argument('-e', type=str, help='题目文件')
    parser.add_argument('-a', type=str, help='答案文件')
    
    args = parser.parse_args()
    generator = MathProblemGenerator()
    
    # 参数验证
    if args.n and args.r:
        # 生成模式
        if args.n <= 0:
            print("错误: -n 必须大于0")
            return
        if args.r <= 1:
            print("错误: -r 必须大于1")
            return
        generator.generate_mode(args.n, args.r)
        
    elif args.e and args.a:
        # 检查模式
        if not os.path.exists(args.e):
            print(f"错误: 题目文件 '{args.e}' 不存在")
            return
        if not os.path.exists(args.a):
            print(f"错误: 答案文件 '{args.a}' 不存在")
            return
        generator.check_mode(args.e, args.a)
        
    else:
        # 参数不完整
        if (args.n and not args.r) or (args.r and not args.n):
            print("错误: -n 和 -r 必须同时使用")
        elif (args.e and not args.a) or (args.a and not args.e):
            print("错误: -e 和 -a 必须同时使用")
        else:
            parser.print_help()

if __name__ == '__main__':
    main()
