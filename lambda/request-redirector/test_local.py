import json
from lambda_function import lambda_handler

def run_test(name, event):
    print(f"--- Running Test: {name} ---")
    try:
        response = lambda_handler(event, None)
        print(f"Status Code: {response.get('statusCode')}")
        body = response.get('body')
        try:
            parsed_body = json.loads(body)
            print(f"Body: {json.dumps(parsed_body, indent=2)[:500]}...", " (truncated)" if len(json.dumps(parsed_body)) > 500 else "")
        except:
            print(f"Body: {body}")
    except Exception as e:
        print(f"Error: {e}")
    print("\n")

if __name__ == "__main__":
    # Test 1: Page Load (No payload)
    run_test("Page Load", {"body": None})

    # Test 2: Valid Location (Denver roughly)
    run_test("Valid Location (Denver)", {
        "body": json.dumps({
            "longitude": -104.99,
            "latitude": 39.73
        })
    })

    # Test 3: Invalid Location (Outside Colorado)
    run_test("Invalid Location (Outside CO)", {
        "body": json.dumps({
            "longitude": -118.24, # LA
            "latitude": 34.05
        })
    })

    # Test 4: Missing Coordinates
    run_test("Missing Coordinates", {
        "body": json.dumps({
            "longitude": -104.99
            # missing latitude
        })
    })
