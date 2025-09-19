# Python Asyncio Made Simple A Complete Tutorial

## When to Use Asyncio, threads, processes

**Asyncio:**

```
Asyncio is a module that provides a way to write concurrent code in a way that is easier to understand
and maintain than using threads or processes.
for managing many waiting tasks.
```
**Threads:**

```
A thread is a light-weight process that runs concurrently with other threads in the same process.
for parallel tasks that share data with minimal cpu use.
```
**Processes:**

```
A process is a heavy-weight process that runs concurrently with other processes in the same operating
system.
for maximizing performance on cpu intensive tasks.
```
## Event Loop

The event loop is a core component of Python's asyncio library, enabling asynchronous programming. It acts
as a central manager, orchestrating the execution of multiple tasks concurrently within a single thread.

```
import asyncio
```
```
# coroutine function
async def main():
print("Start of main coroutine")
```
```
# main() -> coroutine object
```
```
# Run the main coroutine
asyncio.run(main())
```
Await Keyword

```
# Define a coroutine that simulates a time-consuming task
async def fetch_data(delay):
print ("Fetching data...")
await asyncio.sleep(delay) #Simulate an I/0 operation with a sleep
print ("Data fetched")
return {"data": "Some data"} # Return some data
# Define another coroutine that calls the first coroutine
async def main():
```

```
print("Start of main coroutine")
task = fetch_data ( 2 )
# Await the fetch_data coroutine, pausing execution of main until fetch_data
completes
result = await task
print (f"Received result: {result}")
print("End of main coroutine")
```
```
# Run the main coroutine
asyncio.run(main())
```
```
coroutine doesn't start executing until it's awaited
await keyword can be only used inside async function/coroutine
```
## Tasks

A task is a coroutine that is scheduled to run in the event loop. It is created using the
asyncio.create_task() function.

```
import asyncio
```
```
async def fetch_data(id,delay):
print (f"Coroutine {id} starting to fetch data.")
await asyncio.sleep(delay) # Simulate an I/0 operation with a sleep
return {"id":id,"data": "Sample data from coroutine {id}"} # Return some data
```
```
async def main():
task1 = asyncio.create_task(fetch_data( 1 , 2 ))
task2 = asyncio.create_task(fetch_data( 2 , 3 ))
task3 = asyncio.create_task(fetch_data( 3 , 1 ))
```
```
result1 = await task
result2 = await task
result3 = await task
```
```
print(result1,result2,result3)
```
```
# OR we can use
results = await asyncio.gather(task1( 1 , 2 ),task2( 2 , 3 ),task3( 3 , 1 ))
for result in results:
print(f"Recieved result: {result}")
asyncio.run(main())
```
```
If there is error in one of the coroutine and gather is used it will still reamaining coroutines will be
executed.
so we use TaskGroup as it provides some error handling
```

```
import asyncio
```
```
async def fetch_data(id,delay):
print (f"Coroutine {id} starting to fetch data.")
await asyncio.sleep(delay) # Simulate an I/0 operation with a sleep
return {"id":id,"data": "Sample data from coroutine {id}"} # Return some data
```
```
async def main():
tasks =[]
async with asyncio.TaskGroup() as tg:
for i, sleep_time in enumerate([ 2 , 1 , 3 ],start= 1 ):
task = tg.create_task(fetch_data(i,sleep_time))
tasks.append(task)
```
```
results = [task.result() for task in tasks]
for result in results:
print(f"Received result: {result}")
```
```
asyncio.run(main())
```
## Future

Future is a container for a value that will be available in the future.

```
here we are just waiting for a value to be set in the future and not to fully complete the task
because of that in below code we have awaited the future object
```
```
import asyncio
```
```
async def set_future_result(future,value):
await asyncio.sleep( 2 )
```
```
future.set_result(value)
print(f"set the future's result to {value}")
```
```
async def main():
```
```
# Create a future object
loop = asyncio.get_running_loop()
future = loop.create_future()
```
```
# Schedule setting task the future's result
asyncio.create_task(set_future_result(future,"future is ready"))
```
```
result = await future
print(f"Recieved future's result: {result}")
```
```
ayncio.run(main())
```

## Synchronization

Synchronization is the process of ensuring that multiple tasks or processes are executed in a way that avoids
conflicts or overlaps.

```
import asyncio
```
```
shared_resource = 0
```
```
lock = asyncio.Lock()
```
```
async def modify_shared_resource():
global shared_resource
async with lock:
# critical Section starts
print(f"Resource before modification: {shared_resource}")
shared_resource += 1 # modify the shared resource
await asyncio.sleep( 1 ) # simulate an I/O operation
print(f"Resource after modification: {shared_resource}")
# critical Section ends
```
```
async def main():
await asyncio.gather(*(modify_shared_resource() for _ in range( 5 )))
```
```
asyncio.run(main())
```
**Semaphore**

allows multiple coroutines to access a shared resource concurrently, but ensures that only a certain number of
coroutines can access the resource at any given time.

```
import asyncio
```
```
async def access_resource(semaphore,resource_id):
async with semaphore:
# simulate accessing a limited resource
print(f"Accessing resource {resource_id}")
await asyncio.sleep( 1 )
print(f"Releasing resource {resource_id}")
```
```
async def main():
semaphore = asyncio.Semaphore( 2 ) # Allow 2 concurrent access
await asyncio.gather(*(access_resource(semaphore,i) for i in range( 5 )))
```
```
asyncio.run(main())
```
**Event**


Event is used to do simpler synchronization between coroutines.

```
async def waiter(event):
print("waiting for the event to be set")
await event.wait()
print("event has been set, continuing execution")
```
```
async def setter(event):
await asyncio.sleep( 2 )
event.set()
print("event has been set!")
```
```
async def main():
event = asyncio.Event()
await asyncio.gather(waiter(event),setter(event))
```
```
asyncio.run(main())
```
Output:

```
waiting for the event to be set
event has been set!
event has been set, continuing execution
```
**Condition**

soon!


