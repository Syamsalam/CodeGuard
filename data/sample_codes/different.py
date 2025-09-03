import random
import string

def generate_random_string(length):
    """Generate a random string of specified length."""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def bubble_sort(arr):
    """Sort array using bubble sort algorithm."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

class DataProcessor:
    def __init__(self, data=None):
        self.data = data or []
    
    def add_item(self, item):
        self.data.append(item)
    
    def remove_item(self, item):
        if item in self.data:
            self.data.remove(item)
    
    def get_statistics(self):
        if not self.data:
            return {"count": 0, "sum": 0, "avg": 0}
        
        return {
            "count": len(self.data),
            "sum": sum(self.data),
            "avg": sum(self.data) / len(self.data)
        }

def main():
    processor = DataProcessor()
    sample_data = [64, 34, 25, 12, 22, 11, 90]
    
    for item in sample_data:
        processor.add_item(item)
    
    print("Original data:", processor.data)
    sorted_data = bubble_sort(processor.data.copy())
    print("Sorted data:", sorted_data)
    print("Statistics:", processor.get_statistics())
    print("Random string:", generate_random_string(10))

if __name__ == "__main__":
    main()