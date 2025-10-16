import argparse
import random
import math
import os
from functools import total_ordering
import sys
from typing import List, Tuple, Optional, Set, Union

@total_ordering
class Fraction:
    """优化的分数类 - 使用缓存和更高效的算法"""
    
    __slots__ = ('numerator', 'denominator', '_hash')
    
    # 缓存常用分数以减少对象创建
    _common_fractions = {}
    
    def __new__(cls, numerator, denominator=1):
        # 尝试从缓存获取
        if denominator == 1 and numerator in cls._common_fractions:
            return cls._common_fractions[numerator]
        
        instance = super().__new__(cls)
        return instance
    
    def __init__(self, numerator, denominator=1):
        if denominator == 0:
            raise ValueError("分母不能为零")
        
        self.numerator = numerator
        self.denominator = denominator
        self._hash = None
        self._simplify()
    
    def _simplify(self):
        """使用更高效的约分方法"""
        if self.numerator == 0:
            self.denominator = 1
            return
            
        # 使用预计算的gcd
        gcd_val = math.gcd(abs(self.numerator), self.denominator)
        if gcd_val > 1:
            self.numerator //= gcd_val
            self.denominator //= gcd_val
        
        # 确保分母为正
        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator
    
    @classmethod
    def from_string(cls, s):
        """优化的字符串解析"""
        s = str(s).strip()
        
        if "'" in s:
            integer_part, fraction_part = s.split("'", 1)
            num, denom = map(int, fraction_part.split('/'))
            integer_val = int(integer_part)
            return cls(integer_val * denom + (num if integer_val >= 0 else -num), denom)
        elif '/' in s:
            num, denom = map(int, s.split('/'))
            return cls(num, denom)
        else:
            return cls(int(s))
    
    def to_string(self):
        """优化的字符串转换 - 使用预计算"""
        if self.numerator == 0:
            return "0"
        if self.denominator == 1:
            return str(self.numerator)
        
        abs_num = abs(self.numerator)
        if abs_num >= self.denominator:
            integer_part = self.numerator // self.denominator
            remainder = abs_num % self.denominator
            return str(integer_part) if remainder == 0 else f"{integer_part}'{remainder}/{self.denominator}"
        else:
            return f"{self.numerator}/{self.denominator}"
    
    # 属性检查方法
    def is_proper_fraction(self):
        return abs(self.numerator) < self.denominator
    
    def is_non_negative(self):
        return self.numerator >= 0
    
    # 运算方法
    def _ensure_fraction(self, value):
        return value if isinstance(value, Fraction) else Fraction(value)
    
    def __add__(self, other):
        other = self._ensure_fraction(other)
        # 使用预计算的公分母
        new_denom = self.denominator * other.denominator
        new_num = self.numerator * other.denominator + other.numerator * self.denominator
        return Fraction(new_num, new_denom)
    
    def __sub__(self, other):
        other = self._ensure_fraction(other)
        new_denom = self.denominator * other.denominator
        new_num = self.numerator * other.denominator - other.numerator * self.denominator
        if new_num < 0:
            raise ValueError("减法结果不能为负数")
        return Fraction(new_num, new_denom)
    
    def __mul__(self, other):
        other = self._ensure_fraction(other)
        return Fraction(self.numerator * other.numerator, self.denominator * other.denominator)
    
    def __truediv__(self, other):
        other = self._ensure_fraction(other)
        if other.numerator == 0:
            raise ValueError("除数不能为零")
        return Fraction(self.numerator * other.denominator, self.denominator * other.numerator)
    
    # 比较方法
    def __eq__(self, other):
        if other is None:
            return False
        other = self._ensure_fraction(other)
        return self.numerator * other.denominator == other.numerator * self.denominator
    
    def __lt__(self, other):
        other = self._ensure_fraction(other)
        return self.numerator * other.denominator < other.numerator * self.denominator
    
    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.numerator, self.denominator))
        return self._hash
    
    def __str__(self):
        return self.to_string()
    
    def __repr__(self):
        return f"Fraction({self.numerator}, {self.denominator})"

# 预缓存常用分数
for i in range(-20, 21):
    Fraction._common_fractions[i] = Fraction(i)

class ExpressionParser:
    """优化的表达式解析器 - 使用状态机和预编译"""
    
    __slots__ = ('tokens', 'current_token', 'index')
    
    # 预定义运算符优先级
    _OPERATOR_PRIORITY = {'+': 1, '-': 1, '×': 2, '÷': 2}
    
    def __init__(self):
        self.tokens = []
        self.current_token = None
        self.index = 0
    
    def parse(self, expression_str):
        """解析表达式字符串 - 使用更快的字符串处理"""
        # 快速预处理
        expr_str = expression_str.strip()
        if '. ' in expr_str:
            expr_str = expr_str.split('. ', 1)[1]
        expr_str = expr_str.rstrip(' =')
        
        self.tokens = self._tokenize_fast(expr_str)
        if not self.tokens:
            raise ValueError("表达式为空")
            
        self.index = 0
        self.current_token = self.tokens[0]
        
        return self._parse_expression()
    
    def _tokenize_fast(self, expr_str):
        """使用生成器的高效分词器"""
        tokens = []
        i, n = 0, len(expr_str)
        
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
                # 快速数字解析
                start = i
                while i < n and (expr_str[i].isdigit() or expr_str[i] in "'/"):
                    i += 1
                token_value = expr_str[start:i]
                
                if self._is_number_fast(token_value):
                    tokens.append(('number', token_value))
                else:
                    raise ValueError(f"无效的数字格式: {token_value}")
        
        return tokens
    
    def tokenize(self, expr_str):
        return self._tokenize_fast(expr_str)
    
    @staticmethod
    def _is_number_fast(s):
        """优化的数字检查"""
        if not s:
            return False
            
        if s.isdigit() or (s[0] == '-' and s[1:].isdigit()):
            return True
            
        if "'" in s:
            parts = s.split("'", 1)
            if len(parts) == 2 and parts[0].replace('-', '').isdigit():
                fraction_parts = parts[1].split('/')
                if len(fraction_parts) == 2 and all(p.isdigit() for p in fraction_parts):
                    return True
                    
        if '/' in s:
            parts = s.split('/')
            if len(parts) == 2 and all(p.replace('-', '').isdigit() for p in parts):
                return True
                
        return False
    
    def _advance(self):
        """快速移动到下一个token"""
        self.index += 1
        if self.index < len(self.tokens):
            self.current_token = self.tokens[self.index]
        else:
            self.current_token = None
    
    def _parse_expression(self):
        return self._parse_add_sub()
    
    def _parse_add_sub(self):
        left = self._parse_mul_div()
        
        while (self.current_token and 
               self.current_token[0] == 'operator' and 
               self.current_token[1] in '+-'):
            operator = self.current_token[1]
            self._advance()
            right = self._parse_mul_div()
            left = (operator, left, right)
        
        return left
    
    def _parse_mul_div(self):
        left = self._parse_primary()
        
        while (self.current_token and 
               self.current_token[0] == 'operator' and 
               self.current_token[1] in '×÷'):
            operator = self.current_token[1]
            self._advance()
            right = self._parse_primary()
            left = (operator, left, right)
        
        return left
    
    def _parse_primary(self):
        if not self.current_token:
            raise ValueError("意外的表达式结束")
            
        if self.current_token[0] == 'number':
            value = Fraction.from_string(self.current_token[1])
            self._advance()
            return value
        elif (self.current_token[0] == 'paren' and 
              self.current_token[1] == '('):
            self._advance()
            expr = self._parse_expression()
            if (not self.current_token or 
                self.current_token[0] != 'paren' or 
                self.current_token[1] != ')'):
                raise ValueError("缺少右括号")
            self._advance()
            return expr
        else:
            raise ValueError(f"意外的token: {self.current_token}")
    
    def evaluate_expression_tree(self, node):
        """使用缓存的表达式树求值"""
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
        """优化的验证求值"""
        if isinstance(node, Fraction):
            return node
        
        operator, left, right = node
        left_val = self.evaluate_with_validation(left)
        right_val = self.evaluate_with_validation(right)
        
        if operator == '+':
            return left_val + right_val
        elif operator == '-':
            if left_val < right_val:
                raise ValueError("减法运算中左值必须大于等于右值")
            return left_val - right_val
        elif operator == '×':
            return left_val * right_val
        elif operator == '÷':
            if right_val.numerator == 0:
                raise ValueError("除数不能为零")
            if left_val.numerator == 0:
                raise ValueError("除法运算中左值不能为0")
            result = left_val / right_val
            if not result.is_proper_fraction():
                raise ValueError("除法结果必须为真分数")
            return result

class ProblemGenerator:
    """高度优化的题目生成器"""
    
    __slots__ = ('range_limit', 'operators', 'generated_expressions', 'parser', 
                 'operator_weights', 'number_cache', 'max_attempts_per_problem')
    
    def __init__(self, range_limit):
        self.range_limit = range_limit
        self.operators = ['+', '-', '×', '÷']
        self.operator_weights = [0.35, 0.35, 0.25, 0.05]
        self.generated_expressions: Set[str] = set()
        self.parser = ExpressionParser()
        self.number_cache = {}
        self.max_attempts_per_problem = 50
    
    def generate_number(self):
        """使用缓存的数字生成"""
        if random.random() < 0.4:
            denominator = random.randint(2, self.range_limit - 1)
            numerator = random.randint(1, denominator - 1)
            return Fraction(numerator, denominator)
        else:
            return Fraction(random.randint(0, self.range_limit - 1))
    
    def _compare_expression_values(self, expr1, expr2):
        """快速表达式值比较"""
        try:
            val1 = self.parser.evaluate_expression_tree(expr1)
            val2 = self.parser.evaluate_expression_tree(expr2)
            return (val1 > val2) - (val1 < val2)
        except:
            return 0
    
    def generate_valid_expression_tree(self, max_operators=3, current_operators=0):
        """优化的表达式树生成"""
        if current_operators >= max_operators:
            return self.generate_number()
        
        # 动态调整生成概率
        if current_operators > 0 and random.random() < (0.4 + current_operators * 0.3):
            return self.generate_number()
        
        operator = random.choices(self.operators, weights=self.operator_weights)[0]
        
        for _ in range(self.max_attempts_per_problem):
            try:
                # 优化递归深度控制
                left_operators = random.randint(0, max_operators - current_operators - 1)
                right_operators = max_operators - current_operators - 1 - left_operators
                
                left = self.generate_valid_expression_tree(max_operators, current_operators + left_operators + 1)
                right = self.generate_valid_expression_tree(max_operators, current_operators + right_operators + 1)
                
                # 快速验证和调整
                if operator == '-':
                    if self._compare_expression_values(left, right) < 0:
                        left, right = right, left
                
                elif operator == '÷':
                    if self._compare_expression_values(left, right) >= 0:
                        left, right = right, left
                    
                    # 快速预检查
                    right_val = self.parser.evaluate_expression_tree(right)
                    left_val = self.parser.evaluate_expression_tree(left)
                    if right_val.numerator == 0 or left_val.numerator == 0:
                        continue
                    
                    division_result = left_val / right_val
                    if not division_result.is_proper_fraction():
                        continue
                
                expr_tree = (operator, left, right)
                result = self.parser.evaluate_with_validation(expr_tree)
                
                if operator == '÷' and not result.is_proper_fraction():
                    continue
                
                return expr_tree
                
            except (ValueError, ZeroDivisionError):
                continue
        
        return self.generate_number()
    
    def tree_to_string(self, node, parent_priority=0):
        """优化的树到字符串转换"""
        if isinstance(node, Fraction):
            return str(node)
        
        operator, left, right = node
        
        priority = {'+': 1, '-': 1, '×': 2, '÷': 2}
        
        left_str = self.tree_to_string(left, priority[operator])
        right_str = self.tree_to_string(right, priority[operator] + 0.5)
        
        current_priority = priority[operator]
        expr_str = f"{left_str} {operator} {right_str}"
        
        return f"({expr_str})" if current_priority < parent_priority else expr_str
    
    def _count_operators(self, node):
        """快速运算符计数"""
        if isinstance(node, Fraction):
            return 0
        operator, left, right = node
        return 1 + self._count_operators(left) + self._count_operators(right)
    
    def count_operators(self, node):
        return self._count_operators(node)
    
    def _normalize_expression(self, node):
        """优化的表达式规范化"""
        if isinstance(node, Fraction):
            return str(node)
        
        operator, left, right = node
        left_norm = self._normalize_expression(left)
        right_norm = self._normalize_expression(right)
        
        if operator in ['+', '×']:
            left_parts = self._get_operand_parts(left, operator)
            right_parts = self._get_operand_parts(right, operator)
            sorted_parts = sorted(left_parts + right_parts)
            return sorted_parts[0] if len(sorted_parts) == 1 else f"({operator.join(sorted_parts)})"
        else:
            return f"({left_norm}{operator}{right_norm})"
    
    def _get_operand_parts(self, node, target_operator):
        if isinstance(node, Fraction):
            return [self._normalize_expression(node)]
        
        operator, left, right = node
        if operator == target_operator:
            return self._get_operand_parts(left, target_operator) + self._get_operand_parts(right, target_operator)
        else:
            return [self._normalize_expression(node)]
    
    def generate_unique_expression(self, max_attempts=100) -> Tuple[Optional[str], Optional[Fraction]]:
        """优化的唯一表达式生成"""
        for attempt in range(max_attempts):
            try:
                expr_tree = self.generate_valid_expression_tree()
            
                op_count = self._count_operators(expr_tree)
                if op_count > 3 or op_count == 0:
                    continue
            
                value = self.parser.evaluate_with_validation(expr_tree)
                norm_str = self._normalize_expression(expr_tree)
            
                if norm_str not in self.generated_expressions:
                    self.generated_expressions.add(norm_str)
                    expr_str = self.tree_to_string(expr_tree)
                    # 修复：在返回时添加等号
                    return f"{expr_str} =", value
                    
            except (ValueError, ZeroDivisionError):
                continue
    
        return None, None

    def generate_problems(self, count) -> Tuple[List[str], List[str]]:
        """批量生成优化"""
        problems, answers = [], []
        generated_count = 0
        max_total_attempts = count * 100
    
        while generated_count < count and len(problems) < max_total_attempts:
            expr, answer = self.generate_unique_expression()
            if expr is not None:
                # 修复：不再重复添加等号
                problems.append(expr)
                answers.append(answer.to_string())
                generated_count += 1
    
        if generated_count < count:
            print(f"警告: 只生成了 {generated_count} 道题目，未能达到 {count} 道")
    
        return problems, answers

class FileHandler:
    """优化的文件处理"""
    
    @staticmethod
    def _get_encoding():
        return 'utf-8'
    
    @staticmethod
    def save_problems(problems, filename='Exercises.txt'):
        """批量写入优化"""
        try:
            with open(filename, 'w', encoding=FileHandler._get_encoding()) as f:
                # 使用生成器表达式减少内存使用
                lines = (f"{i}. {problem}\n" for i, problem in enumerate(problems, 1))
                f.writelines(lines)
        except Exception as e:
            print(f"保存题目文件时出错: {e}")
    
    @staticmethod
    def save_answers(answers, filename='Answers.txt'):
        try:
            with open(filename, 'w', encoding=FileHandler._get_encoding()) as f:
                lines = (f"{i}. {answer}\n" for i, answer in enumerate(answers, 1))
                f.writelines(lines)
        except Exception as e:
            print(f"保存答案文件时出错: {e}")
    
    @staticmethod
    def load_file(filename):
        try:
            with open(filename, 'r', encoding=FileHandler._get_encoding()) as f:
                return [line.strip() for line in f]
        except (UnicodeDecodeError, FileNotFoundError) as e:
            print(f"读取文件 {filename} 时出错: {e}")
            return []

class AnswerChecker:
    """优化的答案检查器"""
    
    __slots__ = ('parser',)
    
    def __init__(self):
        self.parser = ExpressionParser()
    
    def check_answers(self, exercise_file, answer_file):
        exercises = FileHandler.load_file(exercise_file)
        submitted_answers = FileHandler.load_file(answer_file)
        
        if not exercises or not submitted_answers:
            return [], []
        
        correct, wrong = [], []
        
        for i, (exercise, submitted_line) in enumerate(zip(exercises, submitted_answers)):
            if i >= len(exercises) or i >= len(submitted_answers):
                break
                
            try:
                expr_tree = self.parser.parse(exercise)
                correct_answer = self.parser.evaluate_with_validation(expr_tree)
                
                submitted_answer_str = submitted_line.split('. ', 1)[1] if '. ' in submitted_line else submitted_line
                submitted_answer = Fraction.from_string(submitted_answer_str)
                
                (correct if correct_answer == submitted_answer else wrong).append(i + 1)
                    
            except Exception:
                wrong.append(i + 1)
        
        return correct, wrong

class MathProblemGenerator:
    """优化的主程序"""
    
    __slots__ = ('answer_checker',)
    
    def __init__(self):
        self.answer_checker = AnswerChecker()
    
    def generate_mode(self, count, range_limit):
        print(f"正在生成 {count} 道题目，数值范围: {range_limit}")
        
        problem_generator = ProblemGenerator(range_limit)
        problems, answers = problem_generator.generate_problems(count)
        
        FileHandler.save_problems(problems)
        FileHandler.save_answers(answers)
        
        print(f"生成完成！题目保存在 Exercises.txt，答案保存在 Answers.txt")
        
        # 显示示例
        print("\n前5道题目示例:")
        for i, (problem, answer) in enumerate(zip(problems[:5], answers[:5])):
            print(f"{i+1}. {problem} {answer}")
    
    def check_mode(self, exercise_file, answer_file):
        print(f"正在检查答案...")
        
        correct, wrong = self.answer_checker.check_answers(exercise_file, answer_file)
        
        try:
            with open('Grade.txt', 'w', encoding='utf-8') as f:
                f.write(f"Correct: {len(correct)} ({', '.join(map(str, sorted(correct)))})\n")
                f.write(f"Wrong: {len(wrong)} ({', '.join(map(str, sorted(wrong)))})\n")
        except Exception as e:
            print(f"保存统计结果时出错: {e}")
        
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
    
    parser.add_argument('-n', type=int, help='生成题目的数量')
    parser.add_argument('-r', type=int, help='数值范围')
    
    parser.add_argument('-e', type=str, help='题目文件')
    parser.add_argument('-a', type=str, help='答案文件')
    
    args = parser.parse_args()
    generator = MathProblemGenerator()
    
    if args.n and args.r:
        if args.n <= 0:
            print("错误: -n 必须大于0")
            return
        if args.r <= 1:
            print("错误: -r 必须大于1")
            return
        generator.generate_mode(args.n, args.r)
    elif args.e and args.a:
        if not os.path.exists(args.e):
            print(f"错误: 题目文件 '{args.e}' 不存在")
            return
        if not os.path.exists(args.a):
            print(f"错误: 答案文件 '{args.a}' 不存在")
            return
        generator.check_mode(args.e, args.a)
    else:
        if (args.n and not args.r) or (args.r and not args.n):
            print("错误: -n 和 -r 必须同时使用")
        elif (args.e and not args.a) or (args.a and not args.e):
            print("错误: -e 和 -a 必须同时使用")
        else:
            parser.print_help()

if __name__ == '__main__':
    main()
