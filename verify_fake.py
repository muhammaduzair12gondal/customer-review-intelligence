import requests, time

def test(text):
    t0 = time.time()
    r = requests.post('http://localhost:8000/analyze', json={'text': text})
    res = r.json()
    print(f"Text: '{text}'")
    print(f"  Sentiment: {res['sentiment']} ({res['sentiment_confidence']})")
    print(f"  Fake Prob: {res['is_fake_probability']}")
    print(f"  Time: {time.time()-t0:.2f}s")
    print("-" * 30)

print("Starting verification tests...\n")
# 1. Normal review
test("This is a wonderful product. I highly recommend it to everyone who loves good tea.")
# 2. Suspicious review (short)
test("Bad.")
# 3. Suspicious review (extreme outlier - mock features)
test("A" * 5)
