#!/usr/bin/env python3
# AUTHOR: rendaw
# LICENSE: https://opensource.org/licenses/BSD-2-Clause
# https://github.com/github/markup/issues/1104
import glob
import re
import os.path

for source in glob.glob('./**/*.rst.src', recursive=True):
    dirname = os.path.dirname(source)

    def literalinclude(match):
        with open(os.path.join(dirname, match.group('filename')), 'r') as f:
            body = f.read()

        out = '.. code-block::'
        if match.group('language'):
            out += ' ' + match.group('language')
        out += '\n'

        body = body.splitlines()
        start = match.group('start')
        end = match.group('end')
        if start is not None:
            body = body[int(start) - 1:int(end)]
        deindent = None
        for line in body:
            line_indent = len(re.search('^( *)', line).group(1))
            if deindent is None or line_indent < deindent:
                deindent = line_indent
        out += '\n'
        out += '\n'.join('    ' + line[deindent:] for line in body)

        return out

    dest = re.sub('\\.src', '', source)
    with open(source, 'r') as f:
        text = f.read()
    text = re.sub(
        '^\\.\\. literalinclude:: (?P<filename>.*)$'
        '(?:\\s^(?:'
        '(?:    :language: (?P<language>.*))|'
        '(?:    (?P<linenos>:linenos:))|'
        '(?:    :lines: (?P<start>.*)-(?P<end>.*))'
        ')$)*',
        literalinclude,
        text,
        flags=re.M,
    )
    with open(dest, 'w') as f:
        f.write(text)