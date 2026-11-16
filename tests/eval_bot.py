"""Answer-level regression test for the bot.

Each question carries the fact a correct answer must state and the law it should
cite. Run it against a live server after any change to retrieval, prompts or the
index, and compare the score:

    python3 tests/eval_bot.py                         # http://localhost:10000
    python3 tests/eval_bot.py --url http://host:10000 --only labour

Messages go through the WhatsApp endpoint from numbers starting +000EVAL, so they
appear in /stats as test traffic. Calls are paced to stay inside the Groq free
tier's per-minute token allowance.

The expected facts were taken from the statutes in the corpus; where the law has
since been amended (the death penalty, abolished in 2023) the test asks for the
amended position, because that is what a user needs to hear.
"""
import argparse
import re
import time
import urllib.parse
import urllib.request

# (topic, question, law the answer should name, fact the answer must contain)
CASES = [
    # employment
    ("labour", "How many days of annual leave am I entitled to?", r"labour act", r"fifteen|\b15\b"),
    ("labour", "How long is maternity leave in Ghana?", r"labour act", r"twelve weeks|\b12\b weeks"),
    ("labour", "Can my employer refuse to pay me during maternity leave?", r"labour act", r"full (pay|remuneration)"),
    ("labour", "Is my employer allowed to fire me because I am pregnant?", r"labour act", r"unfair|not allowed|cannot|prohibit"),
    ("labour", "Can I be sacked for joining a trade union?", r"labour act", r"unfair|cannot|not allowed|prohibit"),
    ("labour", "How many hours a day can my employer make me work?", r"labour act", r"eight|\b8\b"),
    ("labour", "How much notice must my employer give before ending my contract of three years or more?", r"labour act", r"one month|1 month"),
    ("labour", "At what age can a child be employed in Ghana?", r"child", r"fifteen|\b15\b"),
    # family
    ("family", "What is the minimum age for marriage?", r"child", r"eighteen|\b18\b"),
    ("family", "Can my parents force me to marry?", r"child|criminal|constitution", r"\bno\b|cannot|not allowed|prohibit|offence"),
    ("family", "How do I get a divorce in Ghana?", r"matrimonial", r"broken down|petition"),
    ("family", "Who gets custody of children after separation?", r"child|matrimonial", r"best interest|welfare"),
    ("family", "My husband beats me, what can I do?", r"domestic violence", r"police|protection order|report"),
    ("family", "What is the punishment for domestic violence?", r"domestic violence", r"two years|fine|penalty units"),
    # inheritance and wills
    ("estate", "My father died without a will, who inherits his house?", r"intestate", r"spouse|wife|children"),
    ("estate", "How do I make a valid will?", r"wills", r"writing|witness|signed"),
    ("estate", "How many witnesses does a will need?", r"wills", r"two|\b2\b"),
    # housing
    ("housing", "How much rent advance can a landlord demand?", r"rent act", r"six months|\b6\b months"),
    ("housing", "Can my landlord evict me without a court order?", r"rent act", r"court|order|magistrate"),
    ("housing", "Is a verbal agreement to sell land binding?", r"conveyanc|land act|contracts", r"writing|written"),
    ("housing", "How do I register my land?", r"land act|lands registry", r"regist"),
    # criminal
    ("criminal", "What is the punishment for stealing?", r"criminal offences", r"second.degree felony"),
    ("criminal", "What is the age of criminal responsibility in Ghana?", r"criminal|juvenile", r"twelve|\b12\b"),
    ("criminal", "What is defilement?", r"criminal offences", r"sixteen|\b16\b"),
    ("criminal", "What is the punishment for rape?", r"criminal offences", r"five years|twenty.five|\b25\b|\b5\b"),
    ("criminal", "What is the punishment for murder in Ghana?", r"criminal", r"life imprisonment|imprisonment for life"),
    ("criminal", "Can the police arrest me without a warrant?", r"procedure|criminal", r"\byes\b|may arrest|can arrest"),
    ("criminal", "How long can police hold me before I see a court?", r"constitution|procedure", r"forty.eight|\b48\b"),
    ("criminal", "Do I have a right to bail?", r"constitution|procedure", r"bail"),
    # rights and civic
    ("rights", "How old must I be to vote in Ghana?", r"constitution|election", r"eighteen|\b18\b"),
    ("rights", "Can a company share my personal data without my consent?", r"data protection", r"consent"),
    ("rights", "What are the rights of a person with disability at work?", r"disab", r"discriminat|employ"),
    ("rights", "Can a child be beaten at school as punishment?", r"child", r"unreasonable|reasonable"),
    # traffic, business, pensions
    ("other", "What happens if I drive without a licence?", r"road traffic", r"offence|fine|penalty"),
    ("other", "What is the penalty for drink driving?", r"road traffic", r"offence|fine|imprisonment"),
    ("other", "How do I register a company?", r"companies act", r"registrar"),
    ("other", "At what age can I start receiving my pension?", r"pension", r"sixty|\b60\b"),
    ("other", "Do I need a lawyer to go to court?", r"legal aid|constitution|courts|civil", r"legal aid|represent|yourself|in person"),
    # citations
    ("cite", "What does Order 1 rule 1 of CI 47 say?", r"civil procedure", r"civil proceedings"),
    ("cite", "What does PNDCL 111 cover?", r"intestate", r"without (a )?will|intestate"),
]


def send(url, user, body):
    data = urllib.parse.urlencode({"From": f"whatsapp:+{user}", "Body": body}).encode()
    raw = urllib.request.urlopen(f"{url}/whatsapp", data, timeout=120).read().decode()
    bodies = re.findall(r"<Body>(.*?)</Body>", raw, re.S)
    return re.sub(r"\s+", " ", bodies[-1]).strip() if bodies else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:10000")
    ap.add_argument("--only", help="run one topic")
    ap.add_argument("--pace", type=float, default=6.0, help="seconds between questions")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.only or c[0] == args.only]
    passed, failures, stamp = 0, [], int(time.time()) % 100000
    for i, (topic, q, law, fact) in enumerate(cases):
        user = f"000EVAL{stamp:05d}{i:02d}"
        send(args.url, user, "hi")
        t0 = time.time()
        answer = send(args.url, user, q)
        secs = time.time() - t0
        body = answer.split("For serious matters")[0]
        law_ok = bool(re.search(law, body, re.I))
        fact_ok = bool(re.search(fact, body, re.I))
        ok = law_ok and fact_ok
        passed += ok
        mark = "PASS" if ok else ("FACT" if law_ok else ("LAW " if fact_ok else "FAIL"))
        print(f"[{mark}] {secs:4.1f}s {topic:8} {q}")
        if not ok:
            failures.append((q, body))
        time.sleep(args.pace)
    print(f"\nscore: {passed}/{len(cases)}")
    for q, body in failures:
        print(f"\n--- {q}\n{body[:420]}")


if __name__ == "__main__":
    main()
