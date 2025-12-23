## ✅ 10 CURL TEST CASES (COPY–PASTE)

## 1️⃣ Legit high-trust domain

```bash
curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \
-H "Content-Type: application/json" \
-d '{"url":"https://google.com"}'
echo
```

## 2️⃣ Legit bank site

```bash
curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \
-H "Content-Type: application/json" \
-d '{"url":"https://www.ecobank.com"}'
echo
```

## 3️⃣ IP-based phishing URL

```bash
curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \
-H "Content-Type: application/json" \
-d '{"url":"http://185.199.108.153/login"}'
echo
```

## 4️⃣ Suspicious .xyz domain

```bash
curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \
-H "Content-Type: application/json" \
-d '{"url":"https://free-prize-winner.xyz"}'
echo
```

## 5️⃣ Credential-harvesting style URL

```bash
curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \
-H "Content-Type: application/json" \
-d '{"url":"https://secure-login-account-update.com"}'
echo
```

## 6️⃣ Long obfuscated URL

```bash
curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \
-H "Content-Type: application/json" \
-d '{"url":"https://login.secure.account.verify.update.user.session.info.com/auth"}'
echo
```

## 7️⃣ HTTPS but scammy wording

```bash
curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \
-H "Content-Type: application/json" \
-d '{"url":"https://secure-bonus-claim-now.net"}'
echo
```

## 8️⃣ URL shortener (high risk)

```bash
curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \
-H "Content-Type: application/json" \
-d '{"url":"https://bit.ly/3xYzAbC"}'
echo
```

## 9️⃣ Ghana MoMo impersonation

```bash
curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \
-H "Content-Type: application/json" \
-d '{"url":"https://momo-secure-gh.com/verify"}'
echo
```

## 🔟 Random benign site

```bash
curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \
-H "Content-Type: application/json" \
-d '{"url":"https://wikipedia.org"}'
echo
```