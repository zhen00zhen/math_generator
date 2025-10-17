import unittest
import tempfile
import os
import time
from math_generator import Fraction, ExpressionParser, FileHandler, AnswerChecker, MathProblemGenerator

class TestFraction(unittest.TestCase):
    """优化的分数测试"""
    
    def test_fraction_creation(self):
        """测试分数创建和简化"""
        test_cases = [
            (5, 1, "5"),
            (3, 4, "3/4"),
            (7, 3, "2'1/3"),
            (4, 8, "1/2"),
            (-3, 4, "-3/4")
        ]
        
        for num, den, expected in test_cases:
            with self.subTest(num=num, den=den):
                f = Fraction(num, den)
                self.assertEqual(str(f), expected)
    
    def test_fraction_from_string(self):
        """测试从字符串创建分数"""
        test_cases = [
            ("5", (5, 1)),
            ("3/4", (3, 4)),
            ("2'1/3", (7, 3)),
            ("-1'1/2", (-3, 2))
        ]
        
        for input_str, expected in test_cases:
            with self.subTest(input=input_str):
                f = Fraction.from_string(input_str)
                self.assertEqual(f.numerator, expected[0])
                self.assertEqual(f.denominator, expected[1])
    
    def test_fraction_operations(self):
        """测试分数运算"""
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 3)
        
        operations = [
            (f1 + f2, "5/6"),
            (f1 - f2, "1/6"),
            (f1 * f2, "1/6"),
            (f1 / f2, "1'1/2")
        ]
        
        for result, expected in operations:
            with self.subTest(operation=expected):
                self.assertEqual(str(result), expected)
    
    def test_fraction_comparison(self):
        """测试分数比较"""
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 3)
        f3 = Fraction(1, 2)
        
        self.assertTrue(f1 > f2)
        self.assertTrue(f2 < f1)
        self.assertTrue(f1 == f3)

class TestExpressionParser(unittest.TestCase):
    """优化的表达式解析器测试"""
    
    def setUp(self):
        self.parser = ExpressionParser()
    
    def test_tokenize(self):
        """测试分词"""
        tokens = self.parser.tokenize("1 + 2 × 3")
        expected = [
            ('number', '1'),
            ('operator', '+'),
            ('number', '2'),
            ('operator', '×'),
            ('number', '3')
        ]
        self.assertEqual(tokens, expected)
    
    def test_parse_expressions(self):
        """批量测试表达式解析"""
        test_cases = [
            ("1 + 2", ('+', Fraction(1), Fraction(2))),
            ("1 + 2 × 3", ('+', Fraction(1), ('×', Fraction(2), Fraction(3)))),
            ("(1 + 2) × 3", ('×', ('+', Fraction(1), Fraction(2)), Fraction(3))),
            ("1/2 + 1/3", ('+', Fraction(1, 2), Fraction(1, 3)))
        ]
        
        for expr, expected in test_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse(expr)
                self.assertEqual(result, expected)
    
    def test_evaluate_expressions(self):
        """批量测试表达式求值"""
        test_cases = [
            ("1 + 2", "3"),
            ("1/2 + 1/3", "5/6"),
            ("(1 + 2) × 3", "9")
        ]
        
        for expr, expected in test_cases:
            with self.subTest(expr=expr):
                expr_tree = self.parser.parse(expr)
                result = self.parser.evaluate_with_validation(expr_tree)
                self.assertEqual(str(result), expected)

class TestProblemGenerator(unittest.TestCase):
    """优化的题目生成器测试"""
    
    def setUp(self):
        self.generator = ProblemGenerator(range_limit=10)
    
    def test_generate_number(self):
        """测试生成数字"""
        for _ in range(50):
            number = self.generator.generate_number()
            self.assertIsInstance(number, Fraction)
            
            if number.denominator > 1:
                self.assertTrue(0 < number.numerator < number.denominator)
                self.assertTrue(number.denominator <= 10)
            else:
                self.assertTrue(0 <= number.numerator < 10)
    
    def test_generate_valid_expression(self):
        """测试生成有效表达式"""
        for _ in range(20):
            try:
                expr_tree = self.generator.generate_valid_expression_tree()
                expr_str = self.generator.tree_to_string(expr_tree)
                
                parser = ExpressionParser()
                parsed_expr = parser.parse(expr_str)
                result = parser.evaluate_with_validation(parsed_expr)
                
                self.assertTrue(result.is_non_negative())
                
                op_count = self.generator.count_operators(expr_tree)
                self.assertTrue(0 <= op_count <= 3)
                
            except Exception as e:
                self.fail(f"生成有效表达式失败: {e}")
    
    def test_expression_uniqueness(self):
        """测试表达式唯一性 - 修复版本"""
        expressions = set()
        for _ in range(50):
            expr, answer = self.generator.generate_unique_expression()
            if expr:
                # 现在表达式应该包含等号
                self.assertIn('=', expr, f"表达式缺少等号: {expr}")
                self.assertNotIn(expr, expressions, f"发现重复表达式: {expr}")
                expressions.add(expr)
                self.assertTrue(len(expr) > 0)

class TestFileHandler(unittest.TestCase):
    """优化的文件处理测试"""
    
    def test_save_and_load_problems(self):
        """测试保存和加载题目"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_exercises.txt")
            test_problems = ["1 + 2 =", "3/4 × 1/2 =", "5 - 3 ="]
            
            FileHandler.save_problems(test_problems, test_file)
            loaded = FileHandler.load_file(test_file)
            
            self.assertEqual(len(loaded), 3)
            self.assertTrue(all(loaded[i].startswith(f"{i+1}. ") for i in range(3)))
    
    def test_save_and_load_answers(self):
        """测试保存和加载答案"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_answers.txt")
            test_answers = ["3", "3/8", "2"]
            
            FileHandler.save_answers(test_answers, test_file)
            loaded = FileHandler.load_file(test_file)
            
            self.assertEqual(len(loaded), 3)
            self.assertTrue(all(loaded[i].startswith(f"{i+1}. ") for i in range(3)))

class TestAnswerChecker(unittest.TestCase):
    """优化的答案检查器测试"""
    
    def setUp(self):
        self.checker = AnswerChecker()
    
    def create_test_files(self, temp_dir, exercises, answers):
        """创建测试文件的辅助方法"""
        exercise_file = os.path.join(temp_dir, "exercises.txt")
        answer_file = os.path.join(temp_dir, "answers.txt")
        
        with open(exercise_file, 'w', encoding='utf-8') as f:
            for i, ex in enumerate(exercises, 1):
                f.write(f"{i}. {ex}\n")
        
        with open(answer_file, 'w', encoding='utf-8') as f:
            for i, ans in enumerate(answers, 1):
                f.write(f"{i}. {ans}\n")
                
        return exercise_file, answer_file
    
    def test_check_correct_answers(self):
        """测试检查正确答案"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exercises = ["1 + 2 =", "1/2 + 1/3 =", "5 - 3 ="]
            answers = ["3", "5/6", "2"]
            
            exercise_file, answer_file = self.create_test_files(temp_dir, exercises, answers)
            correct, wrong = self.checker.check_answers(exercise_file, answer_file)
            
            self.assertEqual(correct, [1, 2, 3])
            self.assertEqual(wrong, [])
    
    def test_check_wrong_answers(self):
        """测试检查错误答案"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exercises = ["1 + 2 =", "1/2 + 1/3 ="]
            answers = ["4", "1/2"]  # 错误答案
            
            exercise_file, answer_file = self.create_test_files(temp_dir, exercises, answers)
            correct, wrong = self.checker.check_answers(exercise_file, answer_file)
            
            self.assertEqual(correct, [])
            self.assertEqual(wrong, [1, 2])

class TestIntegration(unittest.TestCase):
    """优化的集成测试"""
    
    def test_generate_and_check_workflow(self):
        """测试完整的生成和检查流程"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MathProblemGenerator()
            problem_generator = ProblemGenerator(range_limit=10)
            problems, answers = problem_generator.generate_problems(5)
            
            # 验证生成的题目包含等号
            for problem in problems:
                self.assertIn('=', problem, f"题目缺少等号: {problem}")
            
            exercise_file = os.path.join(temp_dir, "Exercises.txt")
            answer_file = os.path.join(temp_dir, "Answers.txt")
            
            FileHandler.save_problems(problems, exercise_file)
            FileHandler.save_answers(answers, answer_file)
            
            self.assertTrue(os.path.exists(exercise_file))
            self.assertTrue(os.path.exists(answer_file))
            
            checker = AnswerChecker()
            correct, wrong = checker.check_answers(exercise_file, answer_file)
            
            self.assertEqual(len(correct), 5)
            self.assertEqual(len(wrong), 0)
    
    def test_expression_validation_rules(self):
        """测试表达式验证规则"""
        parser = ExpressionParser()
        
        invalid_cases = [
            ("1 - 2", "减法结果不能为负数"),
            ("3 ÷ 1", "除法结果必须为真分数"),
            ("1 ÷ 0", "除数不能为零")
        ]
        
        for expr, expected_error in invalid_cases:
            with self.subTest(expr=expr):
                with self.assertRaises(ValueError):
                    expr_tree = parser.parse(expr)
                    parser.evaluate_with_validation(expr_tree)

class TestEdgeCases(unittest.TestCase):
    """优化的边界情况测试"""
    
    def test_zero_handling(self):
        """测试零的处理"""
        f = Fraction(0)
        self.assertEqual(str(f), "0")
        
        parser = ExpressionParser()
        expr = parser.parse("0 + 1/2")
        result = parser.evaluate_with_validation(expr)
        self.assertEqual(str(result), "1/2")
    
    def test_fraction_edge_cases(self):
        """测试分数边界情况"""
        test_cases = [
            (0, 5, "0"),
            (5, 1, "5"),
            (-3, 4, "-3/4"),
            (100, 200, "1/2")
        ]
        
        for num, den, expected in test_cases:
            with self.subTest(num=num, den=den):
                f = Fraction(num, den)
                self.assertEqual(str(f), expected)

class TestProgramRequirements(unittest.TestCase):
    """优化的程序需求测试"""
    
    def test_requirements_compliance(self):
        """测试程序是否符合所有需求"""
        generator = ProblemGenerator(range_limit=10)
        problems, answers = generator.generate_problems(10)
        
        self.assertEqual(len(problems), 10)
        self.assertEqual(len(answers), 10)
        
        parser = ExpressionParser()
        
        for i, (problem, answer) in enumerate(zip(problems, answers)):
            with self.subTest(problem=i+1):
                # 验证题目格式
                self.assertIn('=', problem, f"题目 {i+1} 缺少等号")
                
                # 解析并验证题目
                expr_tree = parser.parse(problem)
                calculated_answer = parser.evaluate_with_validation(expr_tree)
                
                # 检查答案正确性
                self.assertEqual(str(calculated_answer), answer)
                
                # 检查运算符数量
                op_count = problem.count('+') + problem.count('-') + problem.count('×') + problem.count('÷')
                self.assertTrue(op_count <= 3, f"题目 {i+1} 运算符超过3个: {problem}")
                
                # 检查结果非负
                self.assertTrue(calculated_answer.is_non_negative())

def run_performance_test():
    """优化的性能测试"""
    import time
    
    print("运行性能测试...")
    
    # 测试生成性能
    start_time = time.time()
    generator = ProblemGenerator(range_limit=10)
    problems, answers = generator.generate_problems(100)
    end_time = time.time()
    
    print(f"生成{len(problems)}道题目耗时: {end_time - start_time:.2f}秒")
    
    # 验证生成的题目包含等号
    if problems:
        has_equals = all('=' in problem for problem in problems)
        print(f"所有题目都包含等号: {has_equals}")
    
    # 测试解析性能
    if problems:
        parser = ExpressionParser()
        start_time = time.time()
        
        for problem in problems[:50]:
            expr_tree = parser.parse(problem)
            result = parser.evaluate_with_validation(expr_tree)
        
        end_time = time.time()
        print(f"解析50道题目耗时: {end_time - start_time:.2f}秒")

if __name__ == '__main__':
    print("开始优化测试...")
    print("=" * 50)
    
    # 运行单元测试
    unittest.main(verbosity=2, exit=False)
    
    # 运行性能测试
    print("\n" + "=" * 50)
    run_performance_test()
    
    print("\n测试完成！")
