#!/bin/bash

# Production Deployment Script for Kora RAG Service
# This script deploys the RAG service to production

set -e

echo "🚀 Starting Production Deployment for Kora RAG Service..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "rag_agent_supabase.py" ]; then
    print_error "Please run this script from the ragproject directory"
    exit 1
fi

# Step 1: Validate environment variables
print_status "Validating environment variables..."
required_vars=(
    "SUPABASE_URL"
    "SUPABASE_SERVICE_KEY"
    "POSTGRES_URL"
    "OPENAI_API_KEY"
    "TAVILY_API_KEY"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        print_error "Missing required environment variable: $var"
        exit 1
    fi
done

print_status "All required environment variables are set"

# Step 2: Test local functionality
print_status "Testing local functionality..."
python3 -c "import rag_agent_supabase; print('✅ Import successful')"

# Step 3: Deploy to Render (if using Render)
if [ "$DEPLOY_TO_RENDER" = "true" ]; then
    print_status "Deploying to Render..."
    
    # Check if render CLI is installed
    if ! command -v render &> /dev/null; then
        print_warning "Render CLI not found. Please install it or deploy manually."
        print_status "Manual deployment steps:"
        echo "1. Go to https://render.com"
        echo "2. Create a new Web Service"
        echo "3. Connect your GitHub repository"
        echo "4. Set build command: pip install -r requirements.txt"
        echo "5. Set start command: python rag_agent_supabase.py serve --port \$PORT --host 0.0.0.0"
        echo "6. Add environment variables from your .env file"
    else
        render deploy
    fi
fi

# Step 4: Deploy to Railway (if using Railway)
if [ "$DEPLOY_TO_RAILWAY" = "true" ]; then
    print_status "Deploying to Railway..."
    
    # Check if railway CLI is installed
    if ! command -v railway &> /dev/null; then
        print_warning "Railway CLI not found. Please install it or deploy manually."
        print_status "Manual deployment steps:"
        echo "1. Go to https://railway.app"
        echo "2. Create a new project"
        echo "3. Connect your GitHub repository"
        echo "4. Set environment variables from your .env file"
    else
        railway up
    fi
fi

# Step 5: Update backend configuration
print_status "Updating backend configuration..."
if [ -n "$RAG_SERVICE_URL" ]; then
    print_status "RAG Service URL: $RAG_SERVICE_URL"
    print_status "Please update your backend environment variables:"
    echo "RAG_SERVER_URL=$RAG_SERVICE_URL"
    echo "RAG_SERVER_PORT=443"
else
    print_warning "RAG_SERVICE_URL not set. Please update backend configuration manually."
fi

# Step 6: Health check
print_status "Performing health check..."
if [ -n "$RAG_SERVICE_URL" ]; then
    sleep 30  # Wait for deployment to complete
    
    if curl -f "$RAG_SERVICE_URL/health" > /dev/null 2>&1; then
        print_status "✅ Health check passed!"
    else
        print_warning "Health check failed. Service might still be starting up."
    fi
fi

print_status "🎉 Deployment completed!"
print_status "Next steps:"
echo "1. Update your backend environment variables with the new RAG service URL"
echo "2. Test the integration from your frontend"
echo "3. Monitor the service logs for any issues"
echo "4. Set up monitoring and alerts"

echo ""
print_status "Deployment Summary:"
echo "- RAG Service: $RAG_SERVICE_URL"
echo "- Health Check: $RAG_SERVICE_URL/health"
echo "- Chat Endpoint: $RAG_SERVICE_URL/chat"
