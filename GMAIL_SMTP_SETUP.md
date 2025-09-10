# Gmail SMTP Setup Guide

This guide will help you set up Gmail SMTP for sending email reports from the AI Document Assistant.

## Prerequisites

- A Gmail account
- Two-Factor Authentication (2FA) enabled on your Gmail account

## Step-by-Step Setup

### 1. Enable Two-Factor Authentication

1. Go to your [Google Account settings](https://myaccount.google.com/)
2. Click on "Security" in the left sidebar
3. Under "Signing in to Google", click on "2-Step Verification"
4. Follow the prompts to enable 2FA if not already enabled

### 2. Generate App Password

1. Go to [Google App Passwords](https://myaccount.google.com/apppasswords)
2. You may need to sign in again
3. Select "Mail" from the "Select app" dropdown
4. Select "Other (custom name)" from the "Select device" dropdown
5. Enter a name like "AI Document Assistant" or "LangChain Backend"
6. Click "Generate"
7. **Copy the 16-character password** (it will look like: `abcd efgh ijkl mnop`)

### 3. Update Your .env File

Create a `.env` file in the `backend` directory (copy from `.env.example`) and update these values:

```env
# Email Configuration
EMAIL_ADDRESS=your-gmail-address@gmail.com
EMAIL_PASSWORD=your-16-character-app-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

**Important Notes:**
- Use your actual Gmail address for `EMAIL_ADDRESS`
- Use the 16-character App Password (not your regular Gmail password) for `EMAIL_PASSWORD`
- Remove spaces from the App Password when entering it

### 4. Example Configuration

```env
EMAIL_ADDRESS=john.doe@gmail.com
EMAIL_PASSWORD=abcdefghijklmnop
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### 5. Test the Configuration

1. Restart your backend server: `python main.py`
2. You should see: `INFO:__main__:Email configuration found - email reports will be sent`
3. Try sending a test email report through the application

## Troubleshooting

### Common Issues

1. **"Authentication failed" error**
   - Make sure you're using the App Password, not your regular Gmail password
   - Verify that 2FA is enabled on your Gmail account
   - Check that there are no spaces in the App Password

2. **"Less secure app access" error**
   - This shouldn't happen with App Passwords, but if it does, make sure you're using an App Password
   - Google has disabled "Less secure app access" - you must use App Passwords

3. **Connection timeout**
   - Check your firewall settings
   - Verify SMTP_SERVER and SMTP_PORT are correct
   - Try using port 465 with SSL instead of 587 with TLS

### Alternative Configuration (SSL)

If port 587 doesn't work, try:

```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
```

Note: The application is configured for TLS (port 587). If you need SSL (port 465), you may need to modify the email sending code.

## Security Best Practices

1. **Never commit your .env file to version control**
2. **Use App Passwords instead of your main Gmail password**
3. **Regularly rotate your App Passwords**
4. **Revoke unused App Passwords from your Google Account settings**

## Other Email Providers

While this guide focuses on Gmail, you can also use:

- **Outlook**: `smtp-mail.outlook.com:587`
- **Yahoo**: `smtp.mail.yahoo.com:587`
- **Custom SMTP**: Your provider's SMTP settings

For other providers, you'll need to check their specific SMTP settings and authentication requirements.