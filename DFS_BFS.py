#Stack
'''
1. 스택 자료구조
->선입후출
->입구와 출구가 동일한 형태
ex) 박스 쌓기!

→파이썬에서는 리스트 자료형만으로도 구현이 가능하다

stack = []
stack.append(5)
stack.append(2)
stack.append(3)
stack.append(7)
stack.pop()
stack.append(1)
stack.append(4)
stack.pop()

print(stack[::-1])
print(stack)
'''
#Queue
'''
1. 선입선출
입출구 모두 뚫려있는 터널 같은 형태! 대기열 의미함
or 대기줄

from collections import deque
#시간 복잡도 줄이기 위해 직접 구현하지 않고 deque모듈을 사용한다!
queue = deque
#list의 append와 동일하게 작용, O(1) 즉, 상수시간 

queue.append(5)
queue.append(2)
queue.append(3)
queue.append(7)
#왼쪽 자료 삭제, O(1) 즉, 상수시간 일반적으로 파이썬에서 queue구현할 때, append와 popleft를 사용하는 것이 일반적임.
queue.popleft()
queue.append(1)
queue.append(4)
queue.popleft()
#데이터가 오른쪽에서 들어와 왼쪽으로 나감
print(queue)
queue.reverse()#큐 반대로 구현(왼쪽->오른쪽)
print(queue)

'''

#재귀함수
'''
Recursive Function : 자기 자신을 다시 호출하는 함수
DFS 구현에 자주 사용됨
단순 형태의 예제

def recursive_func():
    print("재귀 함수를 호출")
    recursive_func()

#recursive_func() #최대 재귀 깊이 제한으로 오류 생성됨

재귀 : 컴퓨터 스택 프레임에 함수 저장되기 때문에 일종의 스택 자료구조 안에 함수에 대한 정보 차례대로 담김
-> 함수 마지막부터 호출
재귀함수를 문제 풀이에 사용할 경우, 종료조건을 반드시 명시해야 함
종료조건이 없다면 함수 무한히 호출될 수 있음

def recursive_function(i):
    #100번쨰 호출에서 함수 종료
    if i==100:
        return
    print(i,"번째 재귀함수입니다, 곧",i+1,"번쨰 재귀함수를 호출합니다.")
    recursive_function(i+1)
    #함수 종료 후 스택처럼 반대로 재귀함수 호출함.
    print(i,"번쨰 재귀함수를 종료합니다.")

recursive_function(1)
'''
#팩토리얼 예제
'''
def factorial_iterative(n):
    result = 1
    #1부터 n까지 차례대로 곱하기
    for i in range(1, n+1):
        result *= i
    return result

def factorial_recursive(n):
    if n<=1:
        return 1
    return n*factorial_recursive(n-1)

print("반복적 구현한 팩토리얼 :" , factorial_iterative(5))
print("재귀적 구현한 팩토리얼 :",factorial_recursive(5))
'''
#재귀함수의 효과적 사용
'''
유클리드 호제법 : 최대공약수 구할 때 사용함
A,B의 최대공약수는 A를 B로 나눈 나머지인 R과 B의 최대공약수와 같다

def gcd(a,b):
    if a%b==0:
        return b
    return gcd(b,a%b)

print(gcd(192,162))

수학적 정의된 점화식과 같은 형태를 함수로 작성할 수 있게 해 간결하게 풀 수 있도록 함
모든 재귀함수는 반복문을 이용하여 재귀적으로 가능
특정 문제르 만났을 때, 어떤 방식이 좋은지 잘 생각해야함
컴퓨터가 함수를 연속적으로 호출할 때, 컴퓨터 메모리 내부 스택 프레임에 함수가 쌓이기 때문에,
스택을 사용해야 할 떄, 구현상 스택 라이브러리 대신 재귀함수를 이용하는 경우가 많다.
'''
#DFS
'''
DFS : 깊이를 우선으로 탐색하는 알골지므
스택 자료구조혹은 재귀함수를 이용함
1. 탐색 시작 노트를 스택에 삽입 후 방문처리
2. 스택 최상단 노드에 방문하지 않은 인접한 노드가 하나라도 있다면, 해당 노드를 스택에 넣고 방문처리
 -> 방문하지 않은 인접노드가 없다며느 스택 최상단에서 노드 꺼냄
 3. 2번을 할 수 없을 때까지 계속 실행

def dfs(graph,v,visited):
    visited[v] = True
    print(v,end=' ')
    #현재 노드와 연결되어 있는 노드 재귀적으로 방문
    for i in graph[v]:
        if not visited[i]:
            dfs(graph,i,visited)

graph = [
    #보통 첫 번째를 1이라고 하기 떄문에, 0번쨰 노드 아예 비워둠둠
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]
visited = [False]*9
dfs(graph,1,visited)
'''
#BFS
'''
너비 우선 탐색 : 가까운 노드부터 우선적으로 탐색
큐 자료구조 이용!1. 탐ㅁ색 시작 노드 큐에 삽입
2. 큐에서 노드 꺼낸 뒤, 해당 노드의 인접 노드 중 방문하지 않은 노드를 모두 큐에 삽입 -> 방문처리
3. 2번 반복
특정 조건에서 최단경로에 많이 사용됨!
-> 각 단선의 weight가 모두 동일한 상황에서 최단경로 찾는 문제에 많이 사용됨
'''
from collections import deque

def bfs(graph , start , visited):
    #큐 구현
    queue = deque([start])
    
    #현 노드 방문처리
    visited[start] = True
    
    #큐 빌때까지 반복함
    while queue:
        v = queue.popleft()
        print(v,end=" ")

        #아직 방문하지 않은 원소들 큐에 삽입함
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True

graph = [
    #보통 첫 번째를 1이라고 하기 떄문에, 0번쨰 노드 아예 비워둠둠
    [],
    [2,3,8],
    [1,7],#2번 노드와 인접
    [1,4,5], #3번 노드와 인접...
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]
#각 노드의 방문 정보를 표현하기 위한 리스트
visited = [False]*9
#정의된 bps 호출
bfs(graph,1,visited)