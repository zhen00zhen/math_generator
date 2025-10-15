
import tempfile
import os
import sys
from math_generator import Fraction, ExpressionParser, ProblemGenerator, FileHandler, AnswerChecker, MathProblemGenerator

class TestFraction(unittest.TestCase):
    """测试Fraction类"""
    
    def test_fraction_creation(self):
        """测试分数创建和简化"""
        # 整数
        f1 = Fraction(5)
        self.assertEqual(str(f1), "5")
        
        # 真分数
        f2 = Fraction(3, 4)
        self.assertEqual(str(f2), "3/4")
        
        # 带分数
        f3 = Fraction(7, 3)
        self.assertEqual(str(f3), "2'1/3")
        
        # 约分
        f4 = Fraction(4, 8)
        self.assertEqual(str(f4), "1/2")
        
        # 负分数
        f5 = Fraction(-3, 4)
        self.assertEqual(str(f5), "-3/4")
    
    def test_fraction_from_string(self):
        """测试从字符串创建分数"""
        # 整数
        f1 = Fraction.from_string("5")
        self.assertEqual(f1.numerator, 5)
        self.assertEqual(f1.denominator, 1)
        
        # 真分数
        f2 = Fraction.from_string("3/4")
        self.assertEqual(f2.numerator, 3)
        self.assertEqual(f2.denominator, 4)
        
        # 带分数
        f3 = Fraction.from_string("2'1/3")
        self.assertEqual(f3.numerator, 7)
        self.assertEqual(f3.denominator, 3)
        
        # 负带分数
        f4 = Fraction.from_string("-1'1/2")
        self.assertEqual(f4.numerator, -3)
        self.assertEqual(f4.denominator, 2)
    
    def test_fraction_operations(self):
        """测试分数运算"""
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 3)
        
        # 加法
        result = f1 + f2
        self.assertEqual(str(result), "5/6")
        
        # 减法
        result = f1 - f2
        self.assertEqual(str(result), "1/6")
        
        # 乘法
        result = f1 * f2
        self.assertEqual(str(result), "1/6")
        
        # 除法
        result = f1 / f2
        self.assertEqual(str(result), "1'1/2")
    
    def test_fraction_comparison(self):
        """测试分数比较"""
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 3)
        f3 = Fraction(1, 2)
        
        self.assertTrue(f1 > f2)
        self.assertTrue(f2 < f1)
        self.assertTrue(f1 == f3)

class TestExpressionParser(unittest.TestCase):
    """测试表达式解析器"""
    
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
    
    def test_parse_simple_expression(self):
        """测试解析简单表达式"""
        expr = self.parser.parse("1 + 2")
        expected = ('+', Fraction(1), Fraction(2))
        self.assertEqual(expr, expected)
    
    def test_parse_complex_expression(self):
        """测试解析复杂表达式"""
        expr = self.parser.parse("1 + 2 × 3")
        # 应该解析为 1 + (2 × 3)
        expected = ('+', Fraction(1), ('×', Fraction(2), Fraction(3)))
        self.assertEqual(expr, expected)
    
    def test_parse_with_parentheses(self):
        """测试解析带括号的表达式"""
        expr = self.parser.parse("(1 + 2) × 3")
        expected = ('×', ('+', Fraction(1), Fraction(2)), Fraction(3))
        self.assertEqual(expr, expected)
    
    def test_parse_fraction_expression(self):
        """测试解析分数表达式"""
        expr = self.parser.parse("1/2 + 1/3")
        expected = ('+', Fraction(1, 2), Fraction(1, 3))
        self.assertEqual(expr, expected)
    
    def test_evaluate_expression(self):
        """测试表达式求值"""
        # 简单加法
        expr = self.parser.parse("1 + 2")
        result = self.parser.evaluate_with_validation(expr)
        self.assertEqual(str(result), "3")
        
        # 混合运算
        expr = self.parser.parse("1/2 + 1/3")
        result = self.parser.evaluate_with_validation(expr)
        self.assertEqual(str(result), "5/6")
        
        # 带括号的运算
        expr = self.parser.parse("(1 + 2) × 3")
        result = self.parser.evaluate_with_validation(expr)
        self.assertEqual(str(result), "9")

class TestProblemGenerator(unittest.TestCase):
    """测试题目生成器 - 严格测试，不修改期望"""
    
    def setUp(self):
        self.generator = ProblemGenerator(range_limit=10)
    
    def test_generate_number(self):
        """测试生成数字"""
        for _ in range(100):
            number = self.generator.generate_number()
            self.assertIsInstance(number, Fraction)
            
            # 检查数字是否在范围内
            if number.denominator > 1:  # 分数
                self.assertTrue(0 < number.numerator < number.denominator)
                self.assertTrue(number.denominator <= 10)
            else:  # 整数
                self.assertTrue(0 <= number.numerator < 10)
    
    def test_generate_valid_expression(self):
        """测试生成有效表达式 - 严格测试运算符数量不超过3个"""
        for _ in range(50):
            try:
                expr_tree = self.generator.generate_valid_expression_tree()
                expr_str = self.generator.tree_to_string(expr_tree)
                
                # 解析并验证表达式
                parser = ExpressionParser()
                parsed_expr = parser.parse(expr_str)
                result = parser.evaluate_with_validation(parsed_expr)
                
                # 检查结果是否为非负数
                self.assertTrue(result.is_non_negative())
                
                # 严格检查运算符数量不超过3个
                op_count = self.generator.count_operators(expr_tree)
                self.assertTrue(0 <= op_count <= 3, 
                               f"运算符数量 {op_count} 超出范围，表达式: {expr_str}")
                
            except Exception as e:
                self.fail(f"生成有效表达式失败: {e}")
    
    def test_expression_uniqueness(self):
        """测试表达式唯一性 - 严格测试表达式格式"""
        expressions = set()
        for _ in range(100):
            expr, answer = self.generator.generate_unique_expression()
            if expr:
                # 检查是否重复
                self.assertNotIn(expr, expressions, 
                                f"发现重复表达式: {expr}")
                expressions.add(expr)
                
                # 严格检查表达式格式 - 应该包含等号
                self.assertIn('=', expr, 
                             f"表达式缺少等号: {expr}")
                
                # 检查表达式非空
                self.assertTrue(len(expr) > 0)

class TestFileHandler(unittest.TestCase):
    """测试文件处理类"""
    
    def test_save_and_load_problems(self):
        """测试保存和加载题目"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_exercises.txt")
            test_problems = ["1 + 2", "3/4 × 1/2", "5 - 3"]
            
            # 保存题目
            FileHandler.save_problems(test_problems, test_file)
            
            # 加载题目
            loaded = FileHandler.load_file(test_file)
            
            # 检查加载的内容
            self.assertEqual(len(loaded), 3)
            self.assertTrue(loaded[0].startswith("1. "))
            self.assertTrue(loaded[1].startswith("2. "))
            self.assertTrue(loaded[2].startswith("3. "))
    
    def test_save_and_load_answers(self):
        """测试保存和加载答案"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_answers.txt")
            test_answers = ["3", "3/8", "2"]
            
            # 保存答案
            FileHandler.save_answers(test_answers, test_file)
            
            # 加载答案
            loaded = FileHandler.load_file(test_file)
            
            # 检查加载的内容
            self.assertEqual(len(loaded), 3)
            self.assertTrue(loaded[0].startswith("1. "))
            self.assertTrue(loaded[1].startswith("2. "))
            self.assertTrue(loaded[2].startswith("3. "))

class TestAnswerChecker(unittest.TestCase):
    """测试答案检查器"""
    
    def setUp(self):
        self.checker = AnswerChecker()
    
    def test_check_correct_answers(self):
        """测试检查正确答案"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exercise_file = os.path.join(temp_dir, "exercises.txt")
            answer_file = os.path.join(temp_dir, "answers.txt")
            
            # 创建测试文件
            with open(exercise_file, 'w', encoding='utf-8') as f:
                f.write("1. 1 + 2 =\n")
                f.write("2. 1/2 + 1/3 =\n")
                f.write("3. 5 - 3 =\n")
            
            with open(answer_file, 'w', encoding='utf-8') as f:
                f.write("1. 3\n")
                f.write("2. 5/6\n")
                f.write("3. 2\n")
            
            # 检查答案
            correct, wrong = self.checker.check_answers(exercise_file, answer_file)
            
            # 验证结果
            self.assertEqual(correct, [1, 2, 3])
            self.assertEqual(wrong, [])
    
    def test_check_wrong_answers(self):
        """测试检查错误答案"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exercise_file = os.path.join(temp_dir, "exercises.txt")
            answer_file = os.path.join(temp_dir, "answers.txt")
            
            # 创建测试文件
            with open(exercise_file, 'w', encoding='utf-8') as f:
                f.write("1. 1 + 2 =\n")
                f.write("2. 1/2 + 1/3 =\n")
            
            with open(answer_file, 'w', encoding='utf-8') as f:
                f.write("1. 4\n")  # 错误答案
                f.write("2. 1/2\n")  # 错误答案
            
            # 检查答案
            correct, wrong = self.checker.check_answers(exercise_file, answer_file)
            
            # 验证结果
            self.assertEqual(correct, [])
            self.assertEqual(wrong, [1, 2])

class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_generate_and_check_workflow(self):
        """测试完整的生成和检查流程"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 生成题目
            generator = MathProblemGenerator()
            problem_generator = ProblemGenerator(range_limit=10)
            problems, answers = problem_generator.generate_problems(5)
            
            # 保存题目和答案
            exercise_file = os.path.join(temp_dir, "Exercises.txt")
            answer_file = os.path.join(temp_dir, "Answers.txt")
            
            FileHandler.save_problems(problems, exercise_file)
            FileHandler.save_answers(answers, answer_file)
            
            # 验证文件存在
            self.assertTrue(os.path.exists(exercise_file))
            self.assertTrue(os.path.exists(answer_file))
            
            # 检查答案
            checker = AnswerChecker()
            correct, wrong = checker.check_answers(exercise_file, answer_file)
            
            # 严格检查：所有题目都应该正确
            self.assertEqual(len(correct), 5, 
                           f"正确题目数量不足，正确: {len(correct)}, 总共: {len(problems)}")
            self.assertEqual(len(wrong), 0)
    
    def test_expression_validation_rules(self):
        """测试表达式验证规则"""
        parser = ExpressionParser()
        
        # 测试减法不能产生负数
        with self.assertRaises(ValueError):
            expr = parser.parse("1 - 2")
            parser.evaluate_with_validation(expr)
        
        # 测试除法结果必须为真分数
        with self.assertRaises(ValueError):
            expr = parser.parse("3 ÷ 1")
            parser.evaluate_with_validation(expr)
        
        # 测试除数不能为零
        with self.assertRaises(ValueError):
            expr = parser.parse("1 ÷ 0")
            parser.evaluate_with_validation(expr)

class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def test_zero_handling(self):
        """测试零的处理"""
        # 0的表示
        f = Fraction(0)
        self.assertEqual(str(f), "0")
        
        # 包含0的运算
        parser = ExpressionParser()
        expr = parser.parse("0 + 1/2")
        result = parser.evaluate_with_validation(expr)
        self.assertEqual(str(result), "1/2")
    
    def test_large_number_generation(self):
        """测试大量题目生成 - 严格测试必须生成指定数量"""
        generator = ProblemGenerator(range_limit=10)
        
        # 严格测试：必须生成50道题目
        problems, answers = generator.generate_problems(50)
        
        self.assertEqual(len(problems), 50, 
                        f"未能生成50道题目，只生成了{len(problems)}道")
        self.assertEqual(len(answers), 50)
        
        # 验证所有题目都符合要求
        parser = ExpressionParser()
        for i, problem in enumerate(problems):
            try:
                expr_tree = parser.parse(problem)
                result = parser.evaluate_with_validation(expr_tree)
                
                # 检查结果非负
                self.assertTrue(result.is_non_negative(), 
                               f"题目结果不能为负: {problem} = {result}")
                
                # 检查运算符数量
                op_count = problem.count('+') + problem.count('-') + problem.count('×') + problem.count('÷')
                self.assertTrue(op_count <= 3, 
                               f"题目运算符超过3个: {problem}")
                
            except Exception as e:
                self.fail(f"题目 {i+1} 验证失败: {problem}, 错误: {e}")
    
    def test_fraction_edge_cases(self):
        """测试分数边界情况"""
        # 分子为0
        f1 = Fraction(0, 5)
        self.assertEqual(str(f1), "0")
        
        # 分母为1
        f2 = Fraction(5, 1)
        self.assertEqual(str(f2), "5")
        
        # 负分数
        f3 = Fraction(-3, 4)
        self.assertEqual(str(f3), "-3/4")
        
        # 大分数约分
        f4 = Fraction(100, 200)
        self.assertEqual(str(f4), "1/2")

class TestProgramRequirements(unittest.TestCase):
    """测试程序需求符合性"""
    
    def test_requirements_compliance(self):
        """测试程序是否符合所有需求"""
        generator = ProblemGenerator(range_limit=10)
        
        # 测试生成10道题目
        problems, answers = generator.generate_problems(10)
        
        # 严格检查：必须生成10道题目
        self.assertEqual(len(problems), 10, "必须生成10道题目")
        self.assertEqual(len(answers), 10, "必须生成10个答案")
        
        parser = ExpressionParser()
        
        for i, (problem, answer) in enumerate(zip(problems, answers)):
            print(f"检查题目 {i+1}: {problem} = {answer}")
            
            # 检查题目格式
            self.assertIn('=', problem, "题目必须包含等号")
            
            # 解析并验证题目
            expr_tree = parser.parse(problem)
            calculated_answer = parser.evaluate_with_validation(expr_tree)
            
            # 检查答案正确性
            self.assertEqual(str(calculated_answer), answer, 
                           f"答案不匹配: 计算得{calculated_answer}, 期望{answer}")
            
            # 检查运算符数量不超过3个
            op_count = self._count_operators_in_string(problem)
            self.assertTrue(op_count <= 3, 
                           f"运算符数量超过3个: {problem}")
            
            # 检查结果非负
            self.assertTrue(calculated_answer.is_non_negative(), 
                           f"结果不能为负数: {problem} = {calculated_answer}")
    
    def _count_operators_in_string(self, expr_str):
        """计算字符串中的运算符数量"""
        return expr_str.count('+') + expr_str.count('-') + expr_str.count('×') + expr_str.count('÷')

def run_strict_performance_test():
    """严格的性能测试"""
    import time
    
    print("运行严格性能测试...")
    
    # 测试生成1000道题目的性能 - 必须完成
    start_time = time.time()
    
    generator = ProblemGenerator(range_limit=10)
    problems, answers = generator.generate_problems(1000)
    
    end_time = time.time()
    
    print(f"生成{len(problems)}道题目耗时: {end_time - start_time:.2f}秒")
    
    # 严格检查：必须生成1000道题目
    if len(problems) < 1000:
        print(f"警告: 未能生成1000道题目，只生成了{len(problems)}道")
        print("这表示原程序在大量题目生成方面存在问题")
    
    # 测试解析性能
    if len(problems) > 0:
        parser = ExpressionParser()
        start_time = time.time()
        
        for problem in problems[:100]:  # 测试前100道题目的解析
            expr_tree = parser.parse(problem)
            result = parser.evaluate_with_validation(expr_tree)
        
        end_time = time.time()
        print(f"解析100道题目耗时: {end_time - start_time:.2f}秒")

if __name__ == '__main__':
    print("开始严格测试原程序...")
    print("注意：这些测试旨在暴露原程序的真正问题")
    print("=" * 60)
    
    # 运行单元测试
    unittest.main(verbosity=2, exit=False)
    
    # 运行严格的性能测试
    print("\n" + "=" * 60)
    run_strict_performance_test()
    
    print("\n测试完成！")
    print("任何失败的测试都表示原程序存在需要修复的问题")
