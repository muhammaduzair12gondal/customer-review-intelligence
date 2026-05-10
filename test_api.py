import requests, time
t0 = time.time()
try:
    r = requests.post('http://localhost:8000/analyze', json={'text': 'Good Product, loved it'})
    print('Time:', time.time()-t0)
    print(r.json())
except Exception as e:
    print('Failed:', time.time()-t0, e)
