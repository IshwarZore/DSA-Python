class SimpleQueue:
    def __init__(self):
        self.queue = []
    
    def enqueue(self, item):
        self.queue.append(item)  # Add item to the end of the queue
    
    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)  # Remove the item from the front
        else:
            return "Queue is empty"
    
    def peek(self):
        if not self.is_empty():
            return self.queue[0]  # View the first item without removing it
        else:
            return "Queue is empty"
    
    def is_empty(self):
        return len(self.queue) == 0  # Check if the queue is empty
    
    def size(self):
        return len(self.queue)  # Return the size of the queue

def main():
    q = SimpleQueue()
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    
    print("Queue size:", q.size())
    print("Front of the queue:", q.peek())
    
    print("Dequeue:", q.dequeue())
    print("Queue size after dequeue:", q.size())
    print("Front of the queue after dequeue:", q.peek())

if __name__ == "__main__":
    main()