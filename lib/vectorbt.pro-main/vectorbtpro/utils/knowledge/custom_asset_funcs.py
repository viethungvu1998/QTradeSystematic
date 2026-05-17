# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Custom asset function classes."""

import re

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import flat_merge_dicts
from vectorbtpro.utils.knowledge.base_asset_funcs import AssetFunc

__all__ = []


class ToMarkdownAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.custom_assets.VBTAsset.to_markdown`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "to_markdown"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        root_metadata_key: tp.Optional[tp.Key] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **to_markdown_kwargs,
    ) -> tp.ArgsKwargs:
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        if asset is None:
            from vectorbtpro.utils.knowledge.custom_assets import VBTAsset

            asset = VBTAsset
        root_metadata_key = asset.resolve_setting(root_metadata_key, "root_metadata_key")
        clear_metadata = asset.resolve_setting(clear_metadata, "clear_metadata")
        clear_metadata_kwargs = asset.resolve_setting(clear_metadata_kwargs, "clear_metadata_kwargs", merge=True)
        dump_metadata_kwargs = asset.resolve_setting(dump_metadata_kwargs, "dump_metadata_kwargs", merge=True)
        to_markdown_kwargs = asset.resolve_setting(to_markdown_kwargs, "to_markdown_kwargs", merge=True)

        clear_metadata_kwargs = flat_merge_dicts(dict(target=FindRemoveAssetFunc.is_empty_func), clear_metadata_kwargs)
        _, clear_metadata_kwargs = FindRemoveAssetFunc.prepare(**clear_metadata_kwargs)
        _, dump_metadata_kwargs = DumpAssetFunc.prepare(**dump_metadata_kwargs)
        return (), {
            **dict(
                root_metadata_key=root_metadata_key,
                clear_metadata=clear_metadata,
                clear_metadata_kwargs=clear_metadata_kwargs,
                dump_metadata_kwargs=dump_metadata_kwargs,
            ),
            **to_markdown_kwargs,
        }

    @classmethod
    def to_markdown(cls, text: str, remove_code_title: bool = True, even_indentation: bool = True) -> str:
        """Convert text to Markdown.

        If `remove_code_title` is True, removes `title` attribute from a code block and puts it above it.

        If `even_indentation` is True, makes leading spaces even. For example, 3 leading spaces become 4."""

        markdown = text
        if remove_code_title:

            def _replace_code_block(match):
                language = match.group(1)
                title = match.group(2)
                code = match.group(3)
                if title:
                    title_md = f"**{title}**\n\n"
                else:
                    title_md = ""
                code_md = f"```{language}\n{code}\n```"
                return title_md + code_md

            code_block_pattern = re.compile(r'```(\w+)\s+title="([^"]*)"\s*\n(.*?)\n```', re.DOTALL)
            markdown = code_block_pattern.sub(_replace_code_block, markdown)

        if even_indentation:
            leading_spaces_pattern = re.compile(r"^( +)(?=\S|$|\n)")
            fixed_lines = []
            for line in markdown.splitlines(keepends=True):
                match = leading_spaces_pattern.match(line)
                if match and len(match.group(0)) % 2 != 0:
                    line = " " + line
                fixed_lines.append(line)
            markdown = "".join(fixed_lines)

        return markdown

    @classmethod
    def get_markdown_metadata(
        cls,
        d: dict,
        root_metadata_key: tp.Optional[tp.Key] = None,
        allow_empty: bool = True,
        clear_metadata: bool = True,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> str:
        """Get metadata in Markdown format."""
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        if clear_metadata_kwargs is None:
            clear_metadata_kwargs = {}
        if dump_metadata_kwargs is None:
            dump_metadata_kwargs = {}
        metadata = dict(d)
        if "content" in metadata:
            del metadata["content"]
        if metadata and clear_metadata:
            metadata = FindRemoveAssetFunc.call(metadata, **clear_metadata_kwargs)
        if not metadata and not allow_empty:
            return ""
        if root_metadata_key is not None:
            if not metadata:
                metadata = None
            metadata = {root_metadata_key: metadata}
        text = DumpAssetFunc.call(metadata, **dump_metadata_kwargs)
        return cls.to_markdown(text, **kwargs)

    @classmethod
    def get_markdown_content(cls, d: dict, **kwargs) -> str:
        """Get content in Markdown format."""
        if d["content"] is None:
            return ""
        return cls.to_markdown(d["content"], **kwargs)

    @classmethod
    def call(
        cls,
        d: tp.Any,
        root_metadata_key: tp.Optional[tp.Key] = None,
        clear_metadata: bool = True,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        **to_markdown_kwargs,
    ) -> tp.Any:
        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        markdown_metadata = cls.get_markdown_metadata(
            d,
            root_metadata_key=root_metadata_key,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            **to_markdown_kwargs,
        )
        if markdown_metadata:
            markdown_metadata = "---\n" + markdown_metadata + "\n---"
        markdown_content = cls.get_markdown_content(d, **to_markdown_kwargs)
        if markdown_metadata and markdown_content:
            markdown_content = markdown_metadata + "\n\n" + markdown_content
        elif markdown_metadata:
            markdown_content = markdown_metadata
        return markdown_content


class ToHTMLAssetFunc(ToMarkdownAssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.custom_assets.VBTAsset.to_html`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "to_html"

    @classmethod
    def prepare(
        cls,
        root_metadata_key: tp.Optional[tp.Key] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        format_html_kwargs: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **to_html_kwargs,
    ) -> tp.ArgsKwargs:
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        if asset is None:
            from vectorbtpro.utils.knowledge.custom_assets import VBTAsset

            asset = VBTAsset
        root_metadata_key = asset.resolve_setting(root_metadata_key, "root_metadata_key")
        clear_metadata = asset.resolve_setting(clear_metadata, "clear_metadata")
        clear_metadata_kwargs = asset.resolve_setting(clear_metadata_kwargs, "clear_metadata_kwargs", merge=True)
        dump_metadata_kwargs = asset.resolve_setting(dump_metadata_kwargs, "dump_metadata_kwargs", merge=True)
        to_markdown_kwargs = asset.resolve_setting(to_markdown_kwargs, "to_markdown_kwargs", merge=True)
        format_html_kwargs = asset.resolve_setting(format_html_kwargs, "format_html_kwargs", merge=True)
        to_html_kwargs = asset.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True)

        clear_metadata_kwargs = flat_merge_dicts(dict(target=FindRemoveAssetFunc.is_empty_func), clear_metadata_kwargs)
        _, clear_metadata_kwargs = FindRemoveAssetFunc.prepare(**clear_metadata_kwargs)
        _, dump_metadata_kwargs = DumpAssetFunc.prepare(**dump_metadata_kwargs)
        return (), {
            **dict(
                root_metadata_key=root_metadata_key,
                clear_metadata=clear_metadata,
                clear_metadata_kwargs=clear_metadata_kwargs,
                dump_metadata_kwargs=dump_metadata_kwargs,
                to_markdown_kwargs=to_markdown_kwargs,
                format_html_kwargs=format_html_kwargs,
            ),
            **to_html_kwargs,
        }

    @classmethod
    def resolve_extensions(cls, extensions: tp.List[str]) -> tp.List[str]:
        """Resolve markdown extensions.

        Uses `pymdownx` extensions over native extensions if installed."""
        from vectorbtpro.utils.module_ import check_installed

        filtered_extensions = [ext for ext in extensions if "." not in ext or check_installed(ext.partition(".")[0])]
        ext_set = set(filtered_extensions)
        remove_fenced_code = "fenced_code" in ext_set and "pymdownx.superfences" in ext_set
        remove_codehilite = "codehilite" in ext_set and "pymdownx.highlight" in ext_set
        if remove_fenced_code or remove_codehilite:
            filtered_extensions = [
                ext
                for ext in filtered_extensions
                if not ((ext == "fenced_code" and remove_fenced_code) or (ext == "codehilite" and remove_codehilite))
            ]
        return filtered_extensions

    @classmethod
    def make_links(cls, html: str) -> str:
        """Detect raw URLs in HTML text (p and span elements only) and convert them to links."""
        tag_pattern = re.compile(r"<(p|span)(\s[^>]*)?>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
        url_pattern = re.compile(r'(https?://[^\s<>"\'`]+?)(?=[.,;:!?)\]]*(?:\s|$))', re.IGNORECASE)

        def _replace_urls(match, _url_pattern=url_pattern):
            tag = match.group(1)
            attributes = match.group(2) if match.group(2) else ""
            content = match.group(3)
            parts = re.split(r"(<a\b[^>]*>.*?</a>)", content, flags=re.DOTALL | re.IGNORECASE)
            for i, part in enumerate(parts):
                if not re.match(r"<a\b[^>]*>.*?</a>", part, re.DOTALL | re.IGNORECASE):
                    part = _url_pattern.sub(r'<a href="\1">\1</a>', part)
                    parts[i] = part
            new_content = "".join(parts)
            return f"<{tag}{attributes}>{new_content}</{tag}>"

        return tag_pattern.sub(_replace_urls, html)

    @classmethod
    def to_html(
        cls,
        markdown: str,
        resolve_extensions: bool = True,
        make_links: bool = True,
        **kwargs,
    ) -> str:
        """Convert Markdown to HTML.

        If `resolve_extensions` is True, uses `ToHTMLAssetFunc.resolve_extensions`.

        If `make_links` is True, uses `ToHTMLAssetFunc.make_links`.

        Keyword arguments are passed to `markdown.markdown`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("markdown")
        import markdown as md

        extensions = kwargs.pop("extensions", [])
        if resolve_extensions:
            extensions = cls.resolve_extensions(extensions)
        html = md.markdown(markdown, extensions=extensions, **kwargs)
        if make_links:
            html = cls.make_links(html)
        return html.strip()

    @classmethod
    def get_html_metadata(
        cls,
        d: dict,
        root_metadata_key: tp.Optional[tp.Key] = None,
        allow_empty: bool = True,
        clear_metadata: bool = True,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> str:
        """Get metadata in HTML format."""
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}
        metadata = cls.get_markdown_metadata(
            d,
            root_metadata_key=root_metadata_key,
            allow_empty=allow_empty,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            **to_markdown_kwargs,
        )
        metadata = "```yaml\n" + metadata + "\n```"
        return cls.to_html(metadata, **kwargs)

    @classmethod
    def get_html_content(cls, d: dict, to_markdown_kwargs: tp.KwargsLike = None, **kwargs) -> str:
        """Get content in HTML format."""
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}
        content = cls.get_markdown_content(d, **to_markdown_kwargs)
        return cls.to_html(content, **kwargs)

    @classmethod
    def format_html(
        cls,
        title: tp.Optional[str] = None,
        html_metadata: tp.Optional[str] = None,
        html_content: tp.Optional[str] = None,
        use_pygments: tp.Optional[bool] = None,
        pygments_kwargs: tp.KwargsLike = None,
        style_extras: tp.Optional[tp.MaybeList[str]] = None,
        head_extras: tp.Optional[tp.MaybeList[str]] = None,
        body_extras: tp.Optional[tp.MaybeList[str]] = None,
    ) -> str:
        """Format HTML template.

        If `use_pygments` is True, uses Pygments package for code highlighting. Arguments in
        `pygments_kwargs` are then passed to `pygments.formatters.HtmlFormatter`.

        Use `style_extras` to inject additional CSS rules outside the predefined ones.
        Use `head_extras` to inject additional HTML elements into the `<head>` section, such as meta tags,
        links to external stylesheets, or scripts. Use `body_extras` to inject JavaScript files or inline
        scripts at the end of the `<body>`. All of these arguments can be lists."""
        from vectorbtpro.utils.module_ import check_installed, assert_can_import

        if title is None:
            title = ""
        if html_metadata is None:
            html_metadata = ""
        if html_content is None:
            html_content = ""
        if style_extras is None:
            style_extras = []
        if isinstance(style_extras, str):
            style_extras = [style_extras]
        if not isinstance(style_extras, list):
            style_extras = list(style_extras)
        style_extras = "\n".join(style_extras)
        if head_extras is None:
            head_extras = []
        if isinstance(head_extras, str):
            head_extras = [head_extras]
        if not isinstance(head_extras, list):
            head_extras = list(head_extras)
        head_extras = "\n".join(head_extras)
        if body_extras is None:
            body_extras = []
        if isinstance(body_extras, str):
            body_extras = [body_extras]
        if not isinstance(body_extras, list):
            body_extras = list(body_extras)
        body_extras = "\n".join(body_extras)
        if use_pygments is None:
            use_pygments = check_installed("pygments")
        if use_pygments:
            assert_can_import("pygments")
            from pygments.formatters import HtmlFormatter

            formatter = HtmlFormatter(**pygments_kwargs)
            highlight_css = formatter.get_style_defs(".highlight")
            if style_extras == "":
                style_extras = highlight_css
            else:
                style_extras = highlight_css + "\n" + style_extras
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 40px;
            line-height: 1.6;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #333;
        }}
        pre {{
            background-color: #f8f8f8;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            white-space: pre-wrap;
        }}
        .admonition {{
            background-color: #f9f9f9;
            margin: 20px 0;
            padding: 10px 20px;
            border-left: 5px solid #ccc;
            border-radius: 4px;
        }}
        .admonition > p:first-child {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .admonition.example {{
            background-color: #e7f5ff;
            border-left-color: #339af0;
        }}
        .admonition.hint {{
            background-color: #fff4e6;
            border-left-color: #ffa940;
        }}
        .admonition.important {{
            background-color: #ffe3e3;
            border-left-color: #ff6b6b;
        }}
        .admonition.info {{
            background-color: #e3f2fd;
            border-left-color: #42a5f5;
        }}
        .admonition.note {{
            background-color: #e8f5e9;
            border-left-color: #66bb6a;
        }}
        .admonition.question {{
            background-color: #f3e5f5;
            border-left-color: #ab47bc;
        }}
        .admonition.tip {{
            background-color: #fffde7;
            border-left-color: #ffee58;
        }}
        .admonition.warning {{
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }}
        {style_extras}
    </style>
    {head_extras}
</head>
<body>
    {html_metadata}
    {html_content}
    {body_extras}
</body>
</html>"""

    @classmethod
    def call(
        cls,
        d: tp.Any,
        root_metadata_key: tp.Optional[tp.Key] = None,
        clear_metadata: bool = True,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        format_html_kwargs: tp.KwargsLike = None,
        **to_html_kwargs,
    ) -> tp.Any:
        if not isinstance(d, (dict, list)):
            raise TypeError("Data item must be a dict or a list of dicts")
        if isinstance(d, list):
            html_metadata = []
            for _d in d:
                if not isinstance(_d, dict):
                    raise TypeError("Data item must be a dict or a list of dicts")
                html_metadata.append(
                    cls.get_html_metadata(
                        _d,
                        root_metadata_key=root_metadata_key,
                        clear_metadata=clear_metadata,
                        clear_metadata_kwargs=clear_metadata_kwargs,
                        dump_metadata_kwargs=dump_metadata_kwargs,
                        to_markdown_kwargs=to_markdown_kwargs,
                        **to_html_kwargs,
                    )
                )
            html = cls.format_html(
                title="/",
                html_metadata="\n".join(html_metadata),
                **format_html_kwargs,
            )
        else:
            html_metadata = cls.get_html_metadata(
                d,
                root_metadata_key=root_metadata_key,
                clear_metadata=clear_metadata,
                clear_metadata_kwargs=clear_metadata_kwargs,
                dump_metadata_kwargs=dump_metadata_kwargs,
                to_markdown_kwargs=to_markdown_kwargs,
                **to_html_kwargs,
            )
            html_content = cls.get_html_content(
                d,
                to_markdown_kwargs=to_markdown_kwargs,
                **to_html_kwargs,
            )
            html = cls.format_html(
                title=d["link"],
                html_metadata=html_metadata,
                html_content=html_content,
                **format_html_kwargs,
            )
        return html


class AggMessageAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.aggregate_messages`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "agg_message"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        if asset is None:
            from vectorbtpro.utils.knowledge.custom_assets import MessagesAsset

            asset = MessagesAsset
        metadata_format = asset.resolve_setting(metadata_format, "metadata_format")
        clear_metadata = asset.resolve_setting(clear_metadata, "clear_metadata")
        clear_metadata_kwargs = asset.resolve_setting(clear_metadata_kwargs, "clear_metadata_kwargs", merge=True)
        dump_metadata_kwargs = asset.resolve_setting(dump_metadata_kwargs, "dump_metadata_kwargs", merge=True)
        to_html_kwargs = asset.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True)

        clear_metadata_kwargs = flat_merge_dicts(dict(target=FindRemoveAssetFunc.is_empty_func), clear_metadata_kwargs)
        _, clear_metadata_kwargs = FindRemoveAssetFunc.prepare(**clear_metadata_kwargs)
        _, dump_metadata_kwargs = DumpAssetFunc.prepare(**dump_metadata_kwargs)
        return (), {
            **dict(
                metadata_format=metadata_format,
                clear_metadata=clear_metadata,
                clear_metadata_kwargs=clear_metadata_kwargs,
                dump_metadata_kwargs=dump_metadata_kwargs,
                to_html_kwargs=to_html_kwargs,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        metadata_format: str = "markdown",
        clear_metadata: bool = True,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
    ) -> tp.Any:
        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        if clear_metadata_kwargs is None:
            clear_metadata_kwargs = {}
        if dump_metadata_kwargs is None:
            dump_metadata_kwargs = {}
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}
        if to_html_kwargs is None:
            to_html_kwargs = {}

        new_d = dict(d)
        new_d["content"] = new_d["content"].strip()
        attachments = new_d.pop("attachments", [])
        for attachment in attachments:
            content = attachment["content"].strip()
            if new_d["content"]:
                new_d["content"] += "\n\n"
            if metadata_format.lower() == "markdown":
                metadata = ToMarkdownAssetFunc.get_markdown_metadata(
                    attachment,
                    root_metadata_key="attachment",
                    allow_empty=not content,
                    clear_metadata=clear_metadata,
                    clear_metadata_kwargs=clear_metadata_kwargs,
                    dump_metadata_kwargs=dump_metadata_kwargs,
                    **to_markdown_kwargs,
                )
                new_d["content"] += "---\n" + metadata + "\n---"
            elif metadata_format.lower() == "html":
                metadata = ToHTMLAssetFunc.get_html_metadata(
                    attachment,
                    root_metadata_key="attachment",
                    allow_empty=not content,
                    clear_metadata=clear_metadata,
                    clear_metadata_kwargs=clear_metadata_kwargs,
                    dump_metadata_kwargs=dump_metadata_kwargs,
                    to_markdown_kwargs=to_markdown_kwargs,
                    **to_html_kwargs,
                )
                new_d["content"] += metadata
            else:
                raise ValueError(f"Invalid metadata format: '{metadata_format}'")
            if content:
                new_d["content"] += "\n\n" + content
        return new_d


class AggBlockAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.aggregate_blocks`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "agg_block"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        if asset is None:
            from vectorbtpro.utils.knowledge.custom_assets import MessagesAsset

            asset = MessagesAsset
        aggregate_fields = asset.resolve_setting(aggregate_fields, "aggregate_fields")
        parent_links_only = asset.resolve_setting(parent_links_only, "parent_links_only")
        metadata_format = asset.resolve_setting(metadata_format, "metadata_format")
        clear_metadata = asset.resolve_setting(clear_metadata, "clear_metadata")
        clear_metadata_kwargs = asset.resolve_setting(clear_metadata_kwargs, "clear_metadata_kwargs", merge=True)
        dump_metadata_kwargs = asset.resolve_setting(dump_metadata_kwargs, "dump_metadata_kwargs", merge=True)
        to_html_kwargs = asset.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True)

        clear_metadata_kwargs = flat_merge_dicts(dict(target=FindRemoveAssetFunc.is_empty_func), clear_metadata_kwargs)
        _, clear_metadata_kwargs = FindRemoveAssetFunc.prepare(**clear_metadata_kwargs)
        _, dump_metadata_kwargs = DumpAssetFunc.prepare(**dump_metadata_kwargs)
        return (), {
            **dict(
                aggregate_fields=aggregate_fields,
                parent_links_only=parent_links_only,
                metadata_format=metadata_format,
                clear_metadata=clear_metadata,
                clear_metadata_kwargs=clear_metadata_kwargs,
                dump_metadata_kwargs=dump_metadata_kwargs,
                to_html_kwargs=to_html_kwargs,
                link_map=link_map,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        aggregate_fields: tp.Union[bool, tp.MaybeSet[str]] = False,
        parent_links_only: bool = True,
        metadata_format: str = "markdown",
        clear_metadata: bool = True,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
    ) -> tp.Any:
        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        if isinstance(aggregate_fields, bool):
            if aggregate_fields:
                aggregate_fields = {"mentions", "attachments", "reactions"}
            else:
                aggregate_fields = set()
        elif isinstance(aggregate_fields, str):
            aggregate_fields = {aggregate_fields}
        elif not isinstance(aggregate_fields, set):
            aggregate_fields = set(aggregate_fields)
        if clear_metadata_kwargs is None:
            clear_metadata_kwargs = {}
        if dump_metadata_kwargs is None:
            dump_metadata_kwargs = {}
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}
        if to_html_kwargs is None:
            to_html_kwargs = {}

        new_d = {}
        metadata_keys = []
        for k, v in d.items():
            if k == "link":
                new_d[k] = d["block"][0]
            if k == "block":
                continue
            if k in {"thread", "channel", "author"}:
                new_d[k] = v[0]
                continue
            if k == "reference" and link_map is not None:
                found_missing = False
                new_v = []
                for _v in v:
                    if _v:
                        if _v in link_map:
                            _v = link_map[_v]["block"]
                        else:
                            found_missing = True
                            break
                    if _v not in new_v:
                        new_v.append(_v)
                if found_missing or len(new_v) > 1:
                    new_d[k] = "?"
                else:
                    new_d[k] = new_v[0]
            if k == "replies" and link_map is not None:
                new_v = []
                for _v in v:
                    for __v in _v:
                        if __v and __v in link_map:
                            __v = link_map[__v]["block"]
                            if __v not in new_v:
                                new_v.append(__v)
                        else:
                            new_v.append("?")
                new_d[k] = new_v
            if k == "content":
                new_d[k] = []
                continue
            if k in aggregate_fields and isinstance(v[0], list):
                new_v = []
                for _v in new_v:
                    for __v in _v:
                        if __v not in new_v:
                            new_v.append(__v)
                new_d[k] = new_v
                continue
            if k == "reactions" and k in aggregate_fields:
                new_d[k] = sum(v)
                continue
            if parent_links_only:
                if k in ("link", "block", "thread", "reference", "replies"):
                    continue
            metadata_keys.append(k)
        if len(metadata_keys) > 0:
            for i in range(len(d[metadata_keys[0]])):
                content = d["content"][i].strip()
                metadata = {}
                for k in metadata_keys:
                    metadata[k] = d[k][i]
                if len(new_d["content"]) > 0:
                    new_d["content"].append("\n\n")
                if metadata_format.lower() == "markdown":
                    metadata = ToMarkdownAssetFunc.get_markdown_metadata(
                        metadata,
                        root_metadata_key="message",
                        allow_empty=not content,
                        clear_metadata=clear_metadata,
                        clear_metadata_kwargs=clear_metadata_kwargs,
                        dump_metadata_kwargs=dump_metadata_kwargs,
                        **to_markdown_kwargs,
                    )
                    new_d["content"].append("---\n" + metadata + "\n---")
                elif metadata_format.lower() == "html":
                    metadata = ToHTMLAssetFunc.get_html_metadata(
                        metadata,
                        root_metadata_key="message",
                        allow_empty=not content,
                        clear_metadata=clear_metadata,
                        clear_metadata_kwargs=clear_metadata_kwargs,
                        dump_metadata_kwargs=dump_metadata_kwargs,
                        to_markdown_kwargs=to_markdown_kwargs,
                        **to_html_kwargs,
                    )
                    new_d["content"].append(metadata)
                else:
                    raise ValueError(f"Invalid metadata format: '{metadata_format}'")
                if content:
                    new_d["content"].append("\n\n" + content)
        new_d["content"] = "".join(new_d["content"])
        return new_d


class AggThreadAssetFunc(AggBlockAssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.aggregate_threads`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "agg_thread"

    @classmethod
    def call(
        cls,
        d: tp.Any,
        aggregate_fields: tp.Union[bool, tp.MaybeSet[str]] = False,
        parent_links_only: bool = True,
        metadata_format: str = "markdown",
        clear_metadata: bool = True,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
    ) -> tp.Any:
        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        if isinstance(aggregate_fields, bool):
            if aggregate_fields:
                aggregate_fields = {"mentions", "attachments", "reactions"}
            else:
                aggregate_fields = set()
        elif isinstance(aggregate_fields, str):
            aggregate_fields = {aggregate_fields}
        elif not isinstance(aggregate_fields, set):
            aggregate_fields = set(aggregate_fields)
        if clear_metadata_kwargs is None:
            clear_metadata_kwargs = {}
        if dump_metadata_kwargs is None:
            dump_metadata_kwargs = {}
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}
        if to_html_kwargs is None:
            to_html_kwargs = {}

        new_d = {}
        metadata_keys = []
        for k, v in d.items():
            if k == "link":
                new_d[k] = d["thread"][0]
            if k == "thread":
                continue
            if k == "channel":
                new_d[k] = v[0]
                continue
            if k == "content":
                new_d[k] = []
                continue
            if k in aggregate_fields and isinstance(v[0], list):
                new_v = []
                for _v in new_v:
                    for __v in _v:
                        if __v not in new_v:
                            new_v.append(__v)
                new_d[k] = new_v
                continue
            if k == "reactions" and k in aggregate_fields:
                new_d[k] = sum(v)
                continue
            if parent_links_only:
                if k in ("link", "block", "thread", "reference", "replies"):
                    continue
            metadata_keys.append(k)
        if len(metadata_keys) > 0:
            for i in range(len(d[metadata_keys[0]])):
                content = d["content"][i].strip()
                metadata = {}
                for k in metadata_keys:
                    metadata[k] = d[k][i]
                if len(new_d["content"]) > 0:
                    new_d["content"].append("\n\n")
                if metadata_format.lower() == "markdown":
                    metadata = ToMarkdownAssetFunc.get_markdown_metadata(
                        metadata,
                        root_metadata_key="message",
                        allow_empty=not content,
                        clear_metadata=clear_metadata,
                        clear_metadata_kwargs=clear_metadata_kwargs,
                        dump_metadata_kwargs=dump_metadata_kwargs,
                        **to_markdown_kwargs,
                    )
                    new_d["content"].append("---\n" + metadata + "\n---")
                elif metadata_format.lower() == "html":
                    metadata = ToHTMLAssetFunc.get_html_metadata(
                        metadata,
                        root_metadata_key="message",
                        allow_empty=not content,
                        clear_metadata=clear_metadata,
                        clear_metadata_kwargs=clear_metadata_kwargs,
                        dump_metadata_kwargs=dump_metadata_kwargs,
                        to_markdown_kwargs=to_markdown_kwargs,
                        **to_html_kwargs,
                    )
                    new_d["content"].append(metadata)
                else:
                    raise ValueError(f"Invalid metadata format: '{metadata_format}'")
                if content:
                    new_d["content"].append("\n\n" + content)
        new_d["content"] = "".join(new_d["content"])
        return new_d


class AggChannelAssetFunc(AggThreadAssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.aggregate_channels`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "agg_channel"

    @classmethod
    def get_channel_link(cls, link: str) -> str:
        """Get channel link from a message link."""
        if link.startswith("$discord/"):
            link = link[len("$discord/") :]
            link_parts = link.split("/")
            channel_id = link_parts[0]
            return "$discord/" + channel_id
        if link.startswith("https://discord.com/channels/"):
            link = link[len("https://discord.com/channels/") :]
            link_parts = link.split("/")
            guild_id = link_parts[0]
            channel_id = link_parts[1]
            return f"https://discord.com/channels/{guild_id}/{channel_id}"
        raise ValueError(f"Invalid link: '{link}'")

    @classmethod
    def call(
        cls,
        d: tp.Any,
        aggregate_fields: tp.Union[bool, tp.MaybeSet[str]] = False,
        parent_links_only: bool = True,
        metadata_format: str = "markdown",
        clear_metadata: bool = True,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
    ) -> tp.Any:
        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        if isinstance(aggregate_fields, bool):
            if aggregate_fields:
                aggregate_fields = {"mentions", "attachments", "reactions"}
            else:
                aggregate_fields = set()
        elif isinstance(aggregate_fields, str):
            aggregate_fields = {aggregate_fields}
        elif not isinstance(aggregate_fields, set):
            aggregate_fields = set(aggregate_fields)
        if clear_metadata_kwargs is None:
            clear_metadata_kwargs = {}
        if dump_metadata_kwargs is None:
            dump_metadata_kwargs = {}
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}
        if to_html_kwargs is None:
            to_html_kwargs = {}

        new_d = {}
        metadata_keys = []
        for k, v in d.items():
            if k == "link":
                new_d[k] = cls.get_channel_link(v[0])
            if k == "channel":
                new_d[k] = v[0]
                continue
            if k == "content":
                new_d[k] = []
                continue
            if k in aggregate_fields and isinstance(v[0], list):
                new_v = []
                for _v in new_v:
                    for __v in _v:
                        if __v not in new_v:
                            new_v.append(__v)
                new_d[k] = new_v
                continue
            if k == "reactions" and k in aggregate_fields:
                new_d[k] = sum(v)
                continue
            if parent_links_only:
                if k in ("link", "block", "thread", "reference", "replies"):
                    continue
            metadata_keys.append(k)
        if len(metadata_keys) > 0:
            for i in range(len(d[metadata_keys[0]])):
                content = d["content"][i].strip()
                metadata = {}
                for k in metadata_keys:
                    metadata[k] = d[k][i]
                if len(new_d["content"]) > 0:
                    new_d["content"].append("\n\n")
                if metadata_format.lower() == "markdown":
                    metadata = ToMarkdownAssetFunc.get_markdown_metadata(
                        metadata,
                        root_metadata_key="message",
                        allow_empty=not content,
                        clear_metadata=clear_metadata,
                        clear_metadata_kwargs=clear_metadata_kwargs,
                        dump_metadata_kwargs=dump_metadata_kwargs,
                        **to_markdown_kwargs,
                    )
                    new_d["content"].append("---\n" + metadata + "\n---")
                elif metadata_format.lower() == "html":
                    metadata = ToHTMLAssetFunc.get_html_metadata(
                        metadata,
                        root_metadata_key="message",
                        allow_empty=not content,
                        clear_metadata=clear_metadata,
                        clear_metadata_kwargs=clear_metadata_kwargs,
                        dump_metadata_kwargs=dump_metadata_kwargs,
                        to_markdown_kwargs=to_markdown_kwargs,
                        **to_html_kwargs,
                    )
                    new_d["content"].append(metadata)
                else:
                    raise ValueError(f"Invalid metadata format: '{metadata_format}'")
                if content:
                    new_d["content"].append("\n\n" + content)
        new_d["content"] = "".join(new_d["content"])
        return new_d
