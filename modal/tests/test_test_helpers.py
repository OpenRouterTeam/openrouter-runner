from tests.common import get_words_from_file


def test_get_words_from_file():
    """Should return the correct number of words from the file, starting at the correct position"""
    prompt = get_words_from_file(n=10)
    assert len(prompt.split()) == 10
    assert prompt == "I lived long enough to see the cure for death;"

    prompt = get_words_from_file()
    assert prompt.startswith("I lived long enough to see the cure for death;")
    assert prompt.endswith(
        "how to\nsubscribe to our email newsletter to hear about new eBooks."
    )

    prompt = get_words_from_file(start_line=1, n=22)
    assert len(prompt.split()) == 22
    assert (
        prompt
        == "Bitchun Society, to learn ten languages; to compose three symphonies; to realize my boyhood dream of taking up residence in Disney World;"
    )

    # Test a different file
    prompt = get_words_from_file(filename="steve_jobs_speech.txt")
    assert prompt.startswith("thank you I'm honored to be")
    assert prompt.endswith("stay hungry stay foolish thank you all very much")

    # It only has one line, so start_line > 0 should be an empty string
    prompt = get_words_from_file(filename="steve_jobs_speech.txt", start_line=1)
    assert prompt == ""

    prompt = get_words_from_file(
        n=5, filename="steve_jobs_speech.txt", start_line=1
    )
    assert prompt == ""
