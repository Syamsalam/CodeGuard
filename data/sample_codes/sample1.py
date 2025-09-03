def factorial(n):
    """Calculate factorial of a number recursively."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """Calculate fibonacci number iteratively."""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
    
    def multiply(self, x, y):
        return x * y
    
    def calculate_square(self, x):
        return x * x

if __name__ == "__main__":
    calc = Calculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"Factorial of 5: {factorial(5)}")
    print(f"Fibonacci of 10: {fibonacci(10)}")