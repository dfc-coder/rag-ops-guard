def handler(event, context):
    del event, context
    return {"statusCode": 501, "body": "AWS runtime adapter is not part of beta.1"}
