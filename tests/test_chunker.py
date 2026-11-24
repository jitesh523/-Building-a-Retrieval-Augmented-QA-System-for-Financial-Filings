"""Regression tests for document chunking."""
import unittest

from src.data.chunker import DocumentChunker, FixedSizeChunker, SemanticChunker


class FixedSizeChunkerTests(unittest.TestCase):
    def test_rejects_nonpositive_chunk_sizes(self):
        for chunk_size in (0, -1):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(ValueError, 'chunk_size'):
                    FixedSizeChunker(chunk_size=chunk_size, overlap=0)

    def test_rejects_overlap_that_skips_tokens_or_prevents_progress(self):
        for overlap in (-1, 2, 3):
            with self.subTest(overlap=overlap):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    FixedSizeChunker(chunk_size=2, overlap=overlap)

    def test_document_chunker_rejects_invalid_fixed_configuration(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            DocumentChunker(strategy='fixed', chunk_size=2, overlap=2)

    def test_valid_overlap_preserves_content_and_metadata(self):
        chunks = FixedSizeChunker(chunk_size=3, overlap=1).chunk(
            'one two three four five six', page_number=7
        )
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ['one two three', 'three four five', 'five six'],
        )
        self.assertEqual([chunk.index for chunk in chunks], [0, 1, 2])
        self.assertEqual([chunk.token_count for chunk in chunks], [3, 3, 2])
        self.assertTrue(all(chunk.page_number == 7 for chunk in chunks))

    def test_zero_and_maximum_overlap_boundaries(self):
        for overlap, expected in (
            (0, ['one two', 'three four', 'five']),
            (1, ['one two', 'two three', 'three four', 'four five']),
        ):
            with self.subTest(overlap=overlap):
                chunks = FixedSizeChunker(chunk_size=2, overlap=overlap).chunk(
                    'one two three four five'
                )
                self.assertEqual([chunk.text for chunk in chunks], expected)
        chunks = FixedSizeChunker(chunk_size=1, overlap=0).chunk('one two')
        self.assertEqual([chunk.text for chunk in chunks], ['one', 'two'])

    def test_default_configuration_and_empty_input(self):
        chunker = FixedSizeChunker()
        tokens = [str(index) for index in range(901)]
        chunks = chunker.chunk(' '.join(tokens))
        self.assertEqual(
            [chunk.text.split() for chunk in chunks],
            [tokens[:500], tokens[400:900], tokens[800:]],
        )
        self.assertEqual(chunker.chunk(' \n\t'), [])


class SemanticChunkerTests(unittest.TestCase):
    def test_long_sentences_respect_limit_without_losing_tokens(self):
        for count in (80, 81, 105):
            with self.subTest(count=count):
                tokens = [f'word{i}' for i in range(count)]
                chunks = SemanticChunker(max_chunk_size=40).chunk(
                    'ITEM 1. Business\n' + ' '.join(tokens), page_number=7
                )
                self.assertEqual(
                    [token for chunk in chunks for token in chunk.text.split()],
                    tokens,
                )
                self.assertTrue(all(0 < c.token_count <= 40 for c in chunks))
                self.assertEqual([c.index for c in chunks], list(range(len(chunks))))
                self.assertTrue(all(c.page_number == 7 for c in chunks))
                self.assertTrue(all(c.section == 'ITEM 1. Business' for c in chunks))

    def test_long_sentence_between_normal_sentences_preserves_order(self):
        text = ' '.join(['before'] * 25) + '. '
        text += ' '.join(['long'] * 81) + '! '
        text += ' '.join(['after'] * 25) + '?'
        chunks = SemanticChunker(max_chunk_size=40).chunk(text)
        self.assertEqual([c.token_count for c in chunks], [25, 40, 40, 26])
        self.assertEqual(' '.join(c.text for c in chunks).split(), text.split())

    def test_normal_sentences_still_group_together(self):
        sentence = ' '.join(['word'] * 20) + '.'
        chunks = SemanticChunker(max_chunk_size=40).chunk(' '.join([sentence] * 3))
        self.assertEqual([c.token_count for c in chunks], [40, 20])
        self.assertEqual(chunks[0].text, ' '.join([sentence] * 2))

    def test_small_sections_and_empty_input_still_filtered(self):
        chunker = SemanticChunker(max_chunk_size=40)
        for text in ('', ' \n\t', 'A small section.'):
            with self.subTest(text=text):
                self.assertEqual(chunker.chunk(text), [])

    def test_rejects_nonpositive_maximum(self):
        for maximum in (0, -1):
            with self.subTest(maximum=maximum):
                with self.assertRaisesRegex(ValueError, 'max_chunk_size'):
                    DocumentChunker(strategy='semantic', max_chunk_size=maximum)


if __name__ == '__main__':
    unittest.main()
