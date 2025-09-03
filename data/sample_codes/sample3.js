function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

function fibonacci(n) {
    if (n <= 1) return n;
    
    let a = 0, b = 1;
    for (let i = 2; i <= n; i++) {
        [a, b] = [b, a + b];
    }
    return b;
}

class Calculator {
    constructor() {
        this.result = 0;
    }
    
    add(x, y) {
        return x + y;
    }
    
    multiply(x, y) {
        return x * y;
    }
    
    square(x) {
        return x * x;
    }
}

// Usage example
const calc = new Calculator();
console.log(`5 + 3 = ${calc.add(5, 3)}`);
console.log(`Factorial of 5: ${factorial(5)}`);
console.log(`Fibonacci of 10: ${fibonacci(10)}`);