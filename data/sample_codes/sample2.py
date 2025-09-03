def compute_factorial(num):
    """Compute factorial using recursion."""
    if num <= 1:
        return 1
    return num * compute_factorial(num - 1)

def get_fibonacci(num):
    """Get fibonacci number using iteration."""
    if num <= 1:
        return num
    
    first, second = 0, 1
    for index in range(2, num + 1):
        first, second = second, first + second
    return second

class MathCalculator:
    def __init__(self):
        self.current_result = 0
    
    def addition(self, a, b):
        return a + b
    
    def multiplication(self, a, b):
        return a * b
    
    def square_number(self, number):
        return number * number

if __name__ == "__main__":
    calculator = MathCalculator()
    print(f"5 + 3 = {calculator.addition(5, 3)}")
    print(f"Factorial of 5: {compute_factorial(5)}")
    print(f"Fibonacci of 10: {get_fibonacci(10)}")