# AWS EC2 Free Tier Deployment Guide

Deploys the bot on a **t2.micro** instance (1 vCPU, 1 GB RAM) — free for 12 months.
Uses **Amazon Linux 2023** (AWS's own OS — lighter, pre-tuned for EC2, one-command Docker install).
The bot runs in Docker behind nginx with HTTPS via Let's Encrypt.

---

## 1. Launch the EC2 Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Choose **Amazon Linux 2023 AMI** (Free Tier eligible — AWS's recommended OS)
3. Instance type: **t2.micro** (or t3.micro if t2 isn't available in your region — both are free tier)
4. Create a new key pair (e.g. `lawyer-bot.pem`) → download it
5. Security Group — allow inbound:
   - **SSH** (port 22) from your IP
   - **HTTP** (port 80) from anywhere (0.0.0.0/0)
   - **HTTPS** (port 443) from anywhere (0.0.0.0/0)
6. Storage: **20 GB gp3** (free tier allows up to 30 GB)
7. Launch the instance

---

## 2. (Optional) Assign an Elastic IP

Without an Elastic IP the instance gets a new public IP each restart.
- EC2 → Elastic IPs → Allocate → Associate with your instance
- Free while the instance is running

---

## 3. Point a Domain at the Instance

Twilio and Telegram require HTTPS, so you need a domain (or use a free one).

**Free options:**
- [DuckDNS](https://www.duckdns.org) — free subdomain like `mybot.duckdns.org`
- [No-IP](https://noip.com) — free subdomain

Point your chosen domain/subdomain A record → your EC2 public IP.

---

## 4. SSH Into the Instance

```bash
chmod 400 lawyer-bot.pem
ssh -i lawyer-bot.pem ec2-user@YOUR_EC2_IP
```

> Amazon Linux uses `ec2-user`, not `ubuntu`.

---

## 5. Add Swap Space (critical for 1 GB RAM)

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 6. Install Docker

Amazon Linux 2023 ships Docker in its own repo — one command:

```bash
sudo dnf update -y
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user
newgrp docker
```

Install Docker Compose and Buildx plugins (latest versions):
```bash
sudo mkdir -p /usr/local/lib/docker/cli-plugins

# Docker Compose
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Buildx (compose build requires >= 0.17.0; system package is often older)
BUILDX_VER=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep '"tag_name"' | cut -d'"' -f4)
sudo curl -SL "https://github.com/docker/buildx/releases/latest/download/buildx-${BUILDX_VER}.linux-amd64" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx

docker compose version && docker buildx version   # verify both
```

---

## 7. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/Lawyer-freind.git
cd Lawyer-freind
```

> The `ghana_law_vectors/` FAISS index is committed to the repo and will be included automatically.

---

## 8. Create the .env File

```bash
cat > .env << 'EOF'
GROQ_API_KEY=your_groq_api_key_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
EOF
```

---

## 9. Build and Start the Bot

```bash
docker compose up -d --build
```

The first build takes ~5 minutes (downloads Python deps + embedding model).

Check it's running:
```bash
docker compose logs -f
curl http://localhost:10000/health
```

---

## 10. Install nginx

```bash
sudo dnf install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

Create the site config:
```bash
sudo nano /etc/nginx/conf.d/lawyer-bot.conf
```

Paste (replacing `YOUR_DOMAIN` with your actual domain):
```nginx
server {
    listen 80;
    server_name lawyerbot.duckdns.org;

    location / {
        proxy_pass         http://127.0.0.1:10000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

> Amazon Linux uses `/etc/nginx/conf.d/` (no `sites-available/sites-enabled/` like Ubuntu).

Reload nginx:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## 11. Enable HTTPS with Let's Encrypt

```bash
sudo dnf install -y python3-pip augeas-libs
sudo python3 -m venv /opt/certbot/
sudo /opt/certbot/bin/pip install --upgrade pip
sudo /opt/certbot/bin/pip install certbot certbot-nginx
sudo ln -s /opt/certbot/bin/certbot /usr/bin/certbot
sudo certbot --nginx -d lawyerbot.duckdns.org
```

Set up auto-renewal with a systemd timer.

> Do **not** use a `/etc/cron.d` entry here — Amazon Linux 2023 ships without cron installed, so it
> silently never runs and the certificate expires after 90 days. Telegram and Twilio both reject
> expired certificates, which takes the bot offline with no error in your own logs.

```bash
sudo tee /etc/systemd/system/certbot-renew.service > /dev/null <<'EOF'
[Unit]
Description=Renew Let's Encrypt certificates

[Service]
Type=oneshot
ExecStart=/opt/certbot/bin/certbot renew --quiet --deploy-hook "systemctl reload nginx"
EOF

sudo tee /etc/systemd/system/certbot-renew.timer > /dev/null <<'EOF'
[Unit]
Description=Run certbot renew twice daily

[Timer]
OnCalendar=*-*-* 00,12:00:00
RandomizedDelaySec=3600
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now certbot-renew.timer
```

Confirm the timer is scheduled (should show a NEXT run time):
```bash
sudo systemctl list-timers certbot-renew.timer
```

Verify HTTPS works:
```bash
curl https://lawyerbot.duckdns.org/health
# Expected: {"status": "ok"}
```

### If the certificate ever expires

Symptom: the bot stops receiving messages but `/health` still works locally on the instance.
Check what Telegram sees:
```bash
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getWebhookInfo"
# "SSL error ... certificate verify failed" means the cert expired
```
Fix:
```bash
sudo certbot renew --force-renewal
sudo systemctl reload nginx
```

---

## 12. Configure Webhooks

### Twilio (WhatsApp)
1. Go to Twilio Console → Messaging → Try it out → Send a WhatsApp message
2. Set the webhook URL to: `https://lawyerbot.duckdns.org/whatsapp`
3. Method: `HTTP POST`

### Telegram
Register the webhook (run once from your local machine or EC2):
```bash
curl "https://api.telegram.org/bot8948555878:AAHUGYrlLbUGybEFB_sahyBX13OYnx_y_xU/setWebhook?url=https://lawyerbot.duckdns.org/telegram"
```

---

## 13. Auto-start on Reboot

Docker Compose handles this via `restart: unless-stopped`.
nginx and Docker both start automatically via systemd (`sudo systemctl enable` was run in step 6/10).

---

## Useful Commands on EC2

```bash
# View live logs
docker compose logs -f

# Restart after code changes
git pull && docker compose up -d --build

# Stop the bot
docker compose down

# Check memory usage
free -h
```

---

## Cost Estimate

| Resource | Free Tier | After 12 months |
|---|---|---|
| t2.micro compute | 750 hrs/mo free | ~$9/mo |
| 20 GB storage | 30 GB free | ~$2/mo |
| Data transfer | 15 GB out free | ~$1/mo |
| Elastic IP | Free while running | Free while running |
| **Total** | **$0** | **~$12/mo** |
