
def calculate_factorial(number):
    """Calculate factorial using recursion."""
    if number <= 1:
        return 1
    else:
        return number * calculate_factorial(number - 1)

def get_fibonacci_number(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    
    previous, current = 0, 1
    for i in range(2, n + 1):
        previous, current = current, previous + current
    return current

class MathOperations:
    def __init__(self):
        self.last_result = 0
    
    def add_numbers(self, a, b):
        result = a + b
        self.last_result = result
        return result
    
    def multiply_numbers(self, a, b):
        result = a * b
        self.last_result = result
        return result
    
    def square_number(self, num):
        result = num * num
        self.last_result = result
        return result

def main():
    math_ops = MathOperations()
    print(f"Addition: {math_ops.add_numbers(5, 3)}")
    print(f"Factorial: {calculate_factorial(5)}")
    print(f"Fibonacci: {get_fibonacci_number(10)}")

if __name__ == "__main__":
    main()
    