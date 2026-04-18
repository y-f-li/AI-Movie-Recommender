from modules.agent import Agent
from modules.config import Config
from modules.testing_suite import TESTING_SUITE, run_test_suite, build_text_report
from modules.paths import ensure_datapath


def main():
    data_root = ensure_datapath()
    configuration = Config(
        kb_path="",
        ratings_path=str(data_root / "ratings"),
        host_url="direct-cli",
        username="direct-cli",
        password="direct-cli",
        artifacts_dir=str(data_root / "runtime_artifacts"),
    )

    agent = Agent(configuration)
    print("Direct CLI. Type 'quit' (or blank line) to exit.")

    while True:
        try:
            message = input("\nQuestion> ").strip()
            if not message or message.lower() in {"q", "quit", "exit"}:
                print("bye.")
                break
            if message.lower() in {"testing", "test", "tests", "t"}:
                results = run_test_suite(agent, TESTING_SUITE)
                print(build_text_report(results))
                continue
            reply = agent.handle_message(message)
            print(reply)
        except KeyboardInterrupt:
            print("\n^C — exiting.")
            break


if __name__ == "__main__":
    main()
