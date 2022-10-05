
import heapq 

class PriorityQueue:

    def __init__(self): #function to initialize class 
        self.heap = [] #initialize an empty queue as  a list
        self.count = 0
        print("Priority Queue just created")        
    
    def isEmpty(self): #function to check if the queue is empty
        if self.count == 0:
            #print("Queue is empty")
            return True
        else:
            return False

    def findItem(self,item,priority): #function to find an item in the list and return its index
        c = -1 #variable to keep track of the item being tested    
        for x in self.heap: #search the heap to find the item 
            c += 1       
            p,i = x
            if i==item:
                return c
        return -1


    # def findItem(self,item,priority): #function to search for an item in heap
    #     """returned 2 = item found and its priority is smaller than the one we want to update"
    #         returned 1 = item found but its priority is greater, do nothing
    #         returned 0 = item not found """
    #     c = -1 #variable to keep track of the item being tested if match to the one we want to update   
    #     for x in self.heap: #search the heap to find the item
    #         c +=1         
    #         p,i = x
    #         if i==item:
    #             if p>priority: 
    #                 self.heap.pop(c) #delete existing item by index using list pop() function as our heap is implemented as a list
    #                 heapq.heapify(self.heap) #retranform list in a heap after removing the item
    #                 return 2
    #             else:
    #                 return 1
    #     return 0

    def push(self,item,priority): #function to push a new item in the queue 
        if self.findItem(item,priority) == -1:
            heapq.heappush(self.heap,(priority,item)) #from module pq, add an item onte the heap maintaining the heap
            self.count +=1
        else:
            print('Item %s already exist' % item)

    
    def pop(self): #function to delete an item from the queue
        self.count =-1
        if self.isEmpty() == False: 
            return heapq.heappop(self.heap)[-1] #from module pq, return the smallest item from the heap, maintaining the heap
    
    
    def update(self,item,priority): #function to update the priority of an item in the queue
        ret = self.findItem(item,priority)
        if ret != -1:
            p,i = self.heap[ret]
            if p>priority:
                self.heap[ret] = priority,item
                heapq.heapify(self.heap)
        else:
            heapq.heappush(self.heap,(priority,item)) #just add the new item
            self.count +=1

        
        # if ret == 2: #the item is found and deleted
        #     heapq.heappush(self.heap,(priority,item)) #push the new item
        # elif ret == 0: #in case the item we want to update is not in the heap 
        #     heapq.heappush(self.heap,(priority,item)) #just add the new item
        #     self.count +=1


def PQSort(list): #function to sort in increasing order a list of elements(intigers)
    orderedList = []
    q = PriorityQueue() #create an empty heap 
    for x in list: #traverse the list and add each element on the heap
        q.push(x,x)
    for y in range(q.count): #pop elements from the heap and inserts them in a list
        orderedList.append(q.pop())
    return orderedList
        
