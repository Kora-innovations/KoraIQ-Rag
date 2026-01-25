#!/bin/bash
# Quick diagnostic script to test RAG service deployment

echo "🔍 RAG Service Diagnostic Test"
echo "================================"
echo ""

# Replace with your actual RAG service URL
RAG_SERVICE_URL="${1:-https://your-rag-service-name.onrender.com}"

echo "Testing RAG Service: $RAG_SERVICE_URL"
echo ""

# Test 1: Health Check
echo "1️⃣ Testing Health Endpoint..."
HEALTH_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" "$RAG_SERVICE_URL/health")
HTTP_STATUS=$(echo "$HEALTH_RESPONSE" | grep "HTTP_STATUS" | cut -d: -f2)
BODY=$(echo "$HEALTH_RESPONSE" | sed '/HTTP_STATUS/d')

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ Health check passed!"
    echo "Response: $BODY"
else
    echo "❌ Health check failed! HTTP Status: $HTTP_STATUS"
    echo "Response: $BODY"
fi

echo ""
echo "2️⃣ Testing Chat Endpoint..."
CHAT_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "$RAG_SERVICE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test123", "message": "hello"}')
HTTP_STATUS=$(echo "$CHAT_RESPONSE" | grep "HTTP_STATUS" | cut -d: -f2)
BODY=$(echo "$CHAT_RESPONSE" | sed '/HTTP_STATUS/d')

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ Chat endpoint working!"
    echo "Response: $BODY"
else
    echo "❌ Chat endpoint failed! HTTP Status: $HTTP_STATUS"
    echo "Response: $BODY"
fi

echo ""
echo "================================"
echo "Done! If both tests pass, your RAG service is working."
echo ""
echo "Next: Set RAG_SERVER_URL=$RAG_SERVICE_URL in your backend's Render environment variables"
