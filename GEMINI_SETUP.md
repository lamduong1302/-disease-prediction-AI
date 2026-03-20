# Gemini Integration Setup Guide

This document explains how to set up and enable Google Gemini AI for enhancing disease prediction explanations in Vietnamese.

## Overview

The Gemini integration uses Google's Generative AI API to provide intelligent explanations and personalized recommendations for diabetes risk predictions. The system automatically generates Vietnamese language content based on the user's health metrics.

## Prerequisites

1. **Google Cloud Account**: You need a Google Cloud project with the Generative Language API enabled
2. **API Key**: Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
3. **Internet Connection**: The app needs to connect to Google's API servers

## Setup Steps

### 1. Get Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Select your Google Cloud project (or create a new one)
4. Copy the generated API key

### 2. Configure Environment Variables

Set the following environment variables on your system:

**Windows (Command Prompt)**:
```cmd
set GEMINI_API_KEY=your-api-key-here
set GEMINI_MODEL=gemini-1.5-flash
set GEMINI_ENABLED=1
```

**Windows (PowerShell)**:
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
$env:GEMINI_MODEL="gemini-1.5-flash"
$env:GEMINI_ENABLED="1"
```

**Linux/Mac**:
```bash
export GEMINI_API_KEY="your-api-key-here"
export GEMINI_MODEL="gemini-1.5-flash"
export GEMINI_ENABLED="1"
```

### 3. Verify Configuration

The app will automatically read these environment variables from the `config.py` file:
- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `GEMINI_MODEL`: Model name (default: `gemini-1.5-flash`)
- `GEMINI_ENABLED`: Enable/disable Gemini (default: `1` for enabled)

### 4. Run the Application

```bash
python app.py
```

The app will:
- Load the configuration on startup
- Connect to Gemini API when processing predictions
- Fall back to rule-based explanations if Gemini is unavailable

## Features

### What Gemini Does

1. **Enhanced Explanations**: Provides intelligent, Vietnamese language explanations of diabetes risk factors
2. **Personalized Recommendations**: Generates tailored health recommendations based on the user's metrics
3. **Nutrition Tips**: Offers specific dietary advice relevant to the detected risk level
4. **Context Awareness**: Takes into account all health metrics to provide holistic advice

### How It Works

1. User submits health metrics for prediction
2. ML model predicts diabetes risk
3. System generates rule-based explanations
4. **Gemini Processing** (optional):
   - Receives the prediction, risk level, and factors
   - Generates improved Vietnamese explanations
   - Returns personalized recommendations
5. Results are displayed with both rule-based and Gemini-enhanced content

## Configuration Options

### Model Selection

The default model is `gemini-1.5-flash`, which offers:
- ✅ Fast response times
- ✅ Low cost
- ✅ Good quality for this use case
- ✅ Free tier available

Alternative models:
- `gemini-1.5-pro`: Better quality, slower, higher cost
- `gemini-2.0-flash`: Latest model with enhanced capabilities

To switch models, update the environment variable:
```bash
set GEMINI_MODEL=gemini-1.5-pro
```

### Disable Gemini (Optional)

If you want to use only rule-based explanations:
```bash
set GEMINI_ENABLED=0
```

The app will continue working normally with system-generated explanations.

## Troubleshooting

### Gemini Not Responding

**Issue**: "Gemini chưa trả được kết quả phù hợp" (Gemini couldn't return appropriate results)

**Solutions**:
1. Check internet connection
2. Verify API key is correct: `echo %GEMINI_API_KEY%`
3. Check API quota in Google Cloud Console
4. Ensure `GEMINI_ENABLED=1`

### "Gemini returned empty response text"

**Issue**: API key might be invalid or quota exceeded

**Solutions**:
1. Regenerate API key in Google AI Studio
2. Check if you've exceeded the free tier quota
3. Wait a few minutes and try again

### Invalid JSON Response

**Issue**: "Response missing required keys"

**Solutions**:
1. The system has fallback mechanisms and will still work
2. This usually happens with longer input data
3. Try shorter health descriptions or disable Gemini temporarily

### Connection Timeout

**Issue**: Request takes too long (>25 seconds)

**Solutions**:
1. Check your internet connection
2. Google's server might be slow; try again
3. Increase timeout in `gemini_client.py` if needed (set `timeout_s` parameter)

## API Logging

To enable detailed Gemini API debugging:

Add to your code or environment:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show:
- API request/response details
- JSON parsing information
- Retry attempts and delays
- Error logs

## Pricing and Quotas

**Free Tier**:
- 15 requests per minute
- Limited total daily requests
- Suitable for development and demo use

**Paid Tier**:
- Higher rate limits
- Per-token pricing
- Available through Google Cloud

Check current pricing: [Google AI Pricing](https://ai.google.dev/pricing)

## Best Practices

1. **Security**: Never commit your API key to version control
   - Use environment variables or `.env` files (in `.gitignore`)
   - Never share API keys in logs or error messages

2. **Error Handling**: The app gracefully falls back to rule-based explanations
   - Users always get explanations even if Gemini fails
   - Check logs for API issues

3. **Rate Limiting**: Built-in exponential backoff
   - Automatic retry with 2^n second delays
   - Maximum 2 retry attempts (configurable)

4. **Performance**: Gemini adds 1-3 seconds to prediction time
   - Consider async processing for high-traffic scenarios
   - Cache responses for identical inputs if needed

## Advanced Configuration

### Custom Prompts

Edit the `maybe_call_gemini()` function in `app.py` to customize:
- Explanation style
- Recommendation focus
- Language tone
- Content length

### Retry Configuration

Modify `call_gemini()` in `gemini_client.py`:
```python
# Change max_retries parameter
max_retries=3  # Increase retry attempts
```

## Support

- [Google Generative AI Docs](https://ai.google.dev/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [API Error Handling](https://ai.google.dev/docs/error_handling)

## Next Steps

1. ✅ Set up your API key
2. ✅ Configure environment variables
3. ✅ Test with a sample prediction
4. ✅ Monitor logs for any issues
5. ✅ Customize prompts for your use case (optional)

Enjoy enhanced AI-powered health explanations! 🚀
