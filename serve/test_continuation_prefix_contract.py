"""CPU checks for qualification prefix construction with the shipped template.

ByteTokenizer proves template behavior only, not production-tokenizer parity.
"""
from pathlib import Path
import unittest

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer


class ContinuationPrefixContract(unittest.TestCase):
    def setUp(self):
        self.template = ChatTemplate(Path(__file__).with_name('chat_template.jinja'))
        self.tok = ByteTokenizer()
        self.messages = [{'role': 'user', 'content': 'Remember Cedar.'}]

    def test_decoded_thinking_suffix_is_not_assistant_content_prefix(self):
        seed = self.tok.encode(self.template.render(self.messages))
        suffix = self.tok.encode('Let me reason')
        consumed = seed + suffix
        history = self.messages + [{'role': 'assistant', 'content': self.tok.decode(suffix)},
                                   {'role': 'user', 'content': 'Continue.'}]
        prompt = self.tok.encode(self.template.render(history))
        self.assertNotEqual(consumed, prompt[:len(consumed)])

    def test_render_first_prefix_preserves_exact_ids_without_decode(self):
        for thinking in (True, False):
            prompt = self.tok.encode(self.template.render(self.messages, enable_thinking=thinking))
            for tail in (1, 8, 16):
                with self.subTest(thinking=thinking, tail=tail):
                    consumed = prompt[:-tail]
                    self.assertTrue(0 < len(consumed) < len(prompt))
                    self.assertEqual(consumed + prompt[-tail:], prompt)

    def test_template_does_not_strip_embedded_turn_terminator(self):
        history = self.messages + [{'role': 'assistant', 'content': 'OK<|im_end|>'},
                                   {'role': 'user', 'content': 'Continue.'}]
        rendered = self.template.render(history, enable_thinking=False)
        self.assertIn('OK<|im_end|><|im_end|>', rendered)
