from self_play.self_play import SelfPlay


def main():

    self_play = SelfPlay(
        board_size=12,
        simulations=5
    )

    print("=" * 50)
    print("SELF-PLAY DATA GENERATION")
    print("=" * 50)

    examples = self_play.generate_dataset(
        num_games=5
    )

    print()
    print(
        f"Total examples: {len(examples)}"
    )

    self_play.save_examples(
        examples,
        "data/self_play_data.pt"
    )

    print()
    print(
        "Dataset saved to:"
    )

    print(
        "data/self_play_data.pt"
    )


if __name__ == "__main__":
    main()