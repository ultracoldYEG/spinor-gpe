"""Modification of code generating a README.rst file from other .rst files.

Run this script before pushing documentation changes.

ORIGINAL AUTHOR: rendaw
YEAR: 2017
LICENSE: https://opensource.org/licenses/BSD-2-Clause
SOURCE: https://github.com/github/markup/issues/1104
"""
import glob
import re
import os.path

for source in glob.glob('./**/*.rst.src', recursive=True):
    dirname = os.path.dirname(source)

    def include(match):
        # Reads a .rst file and returns its contents as a string
        with open(os.path.join(dirname, match.group('filename')), 'r') as f:
            body = f.read()
        return body

    def literalinclude(match):
        # Reads a .rst file and returns its contents as a code block directive
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

    # Substitude contents of file for every '.. include' directive
    text = re.sub(
        '^\\.\\. include:: (?P<filename>.*)$',
        include,
        text,
        flags=re.M,
        )

    # Substitude contents of file for every '.. literalinclude' directive
    # Replace it with a code block
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