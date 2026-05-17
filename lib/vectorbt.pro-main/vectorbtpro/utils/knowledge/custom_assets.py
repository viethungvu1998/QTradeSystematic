# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Custom asset classes."""

import io
import os
import re
from types import ModuleType
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import flat_merge_dicts
from vectorbtpro.utils.module_ import prepare_refname, get_caller_qualname
from vectorbtpro.utils.path_ import check_mkdir, remove_dir, get_common_prefix, dir_tree_from_paths
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.pickling import suggest_compression
from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

__all__ = [
    "VBTAsset",
    "PagesAsset",
    "MessagesAsset",
]


VBTAssetT = tp.TypeVar("VBTAssetT", bound="VBTAsset")


class VBTAsset(KnowledgeAsset):
    """Class for working with VBT content.

    For defaults, see `assets.vbt` in `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge.assets.vbt"

    @classmethod
    def pull(
        cls: tp.Type[VBTAssetT],
        asset_name: tp.Optional[str] = None,
        release_name: tp.Optional[str] = None,
        repo_owner: tp.Optional[str] = None,
        repo_name: tp.Optional[str] = None,
        token: tp.Optional[str] = None,
        token_required: tp.Optional[bool] = None,
        use_pygithub: tp.Optional[bool] = None,
        chunk_size: tp.Optional[int] = None,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: bool = False,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> VBTAssetT:
        """Build `VBTAsset` from a JSON asset of a release."""
        from vectorbtpro._version import __version__
        import requests

        asset_name = cls.resolve_setting(asset_name, "asset_name")
        release_name = cls.resolve_setting(release_name, "release_name")
        repo_owner = cls.resolve_setting(repo_owner, "repo_owner")
        repo_name = cls.resolve_setting(repo_name, "repo_name")
        token = cls.resolve_setting(token, "token")
        token_required = cls.resolve_setting(token_required, "token_required")
        use_pygithub = cls.resolve_setting(use_pygithub, "use_pygithub")
        chunk_size = cls.resolve_setting(chunk_size, "chunk_size")
        cache = cls.resolve_setting(cache, "cache")
        cache_dir_none = cache_dir is None
        cache_dir = cls.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = cls.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)
        show_progress = cls.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = cls.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        current_release = "v" + __version__
        if release_name is None:
            release_name = current_release
        release_dir = Path(cache_dir)
        if cache_dir_none:
            release_dir /= "releases"
            release_dir /= release_name
        if cache:
            if release_dir.exists():
                if clear_cache:
                    remove_dir(release_dir, missing_ok=True, with_contents=True)
                else:
                    cache_file = None
                    for file in release_dir.iterdir():
                        if file.is_file() and file.name == asset_name:
                            cache_file = file
                            break
                    if cache_file is not None:
                        return cls.from_json_file(cache_file, **kwargs)

        if token is None:
            token = os.environ.get("GITHUB_TOKEN", None)
        if token is None and token_required:
            raise ValueError("GitHub token is required")
        if use_pygithub is None:
            from vectorbtpro.utils.module_ import check_installed

            use_pygithub = check_installed("github")
        if use_pygithub:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("github")
            from github import Github, Auth
            from github.GithubException import UnknownObjectException

            if token is not None:
                g = Github(auth=Auth.Token(token))
            else:
                g = Github()
            try:
                repo = g.get_repo(f"{repo_owner}/{repo_name}")
            except UnknownObjectException:
                raise Exception(f"Repository '{repo_owner}/{repo_name}' not found or access denied")
            if release_name == "latest":
                try:
                    release = repo.get_latest_release()
                except UnknownObjectException:
                    raise Exception("Latest release not found")
            else:
                releases = repo.get_releases()
                found_release = None
                for release in releases:
                    if release.title == release_name:
                        found_release = release
                if found_release is None:
                    raise Exception(f"Release '{release_name}' not found")
                release = found_release
            assets = release.get_assets()
            if asset_name is not None:
                asset = next((a for a in assets if a.name == asset_name), None)
                if asset is None:
                    raise Exception(f"Asset '{asset_name}' not found in release {release}")
            else:
                assets_list = list(assets)
                if len(assets_list) == 1:
                    asset = assets_list[0]
                else:
                    raise Exception("Please specify asset_name")
            asset_url = asset.url
        else:
            headers = {"Accept": "application/vnd.github+json"}
            if token is not None:
                headers["Authorization"] = f"token {token}"
            if release_name == "latest":
                release_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
                response = requests.get(release_url, headers=headers)
                response.raise_for_status()
                release_info = response.json()
            else:
                releases_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"
                response = requests.get(releases_url, headers=headers)
                response.raise_for_status()
                releases = response.json()
                release_info = None
                for release in releases:
                    if release.get("name") == release_name:
                        release_info = release
                if release_info is None:
                    raise ValueError(f"Release '{release_name}' not found")
            assets = release_info.get("assets", [])
            if asset_name is not None:
                asset = next((a for a in assets if a["name"] == asset_name), None)
                if asset is None:
                    raise Exception(f"Asset '{asset_name}' not found in release {release}")
            else:
                if len(assets) == 1:
                    asset = assets[0]
                else:
                    raise Exception("Please specify asset_name")
            asset_url = asset["url"]

        asset_headers = {"Accept": "application/octet-stream"}
        if token is not None:
            asset_headers["Authorization"] = f"token {token}"
        asset_response = requests.get(asset_url, headers=asset_headers, stream=True)
        asset_response.raise_for_status()
        file_size = int(asset_response.headers.get("Content-Length", 0))
        if file_size == 0:
            file_size = asset.get("size", 0)
        if show_progress is None:
            show_progress = True
        pbar_kwargs = flat_merge_dicts(
            dict(
                bar_id=get_caller_qualname(),
                unit="iB",
                unit_scale=True,
                prefix=f"Downloading {asset_name}",
            ),
            pbar_kwargs,
        )

        if cache:
            check_mkdir(release_dir, **cache_mkdir_kwargs)
            cache_file = release_dir / asset_name
            with open(cache_file, "wb") as f:
                with ProgressBar(total=file_size, show_progress=show_progress, **pbar_kwargs) as pbar:
                    for chunk in asset_response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return cls.from_json_file(cache_file, **kwargs)
        else:
            with io.BytesIO() as bytes_io:
                with ProgressBar(total=file_size, show_progress=show_progress, **pbar_kwargs) as pbar:
                    for chunk in asset_response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            bytes_io.write(chunk)
                            pbar.update(len(chunk))
                bytes_ = bytes_io.getvalue()
            compression = suggest_compression(asset_name)
            if compression is not None and "compression" not in kwargs:
                kwargs["compression"] = compression
            return cls.from_json_bytes(bytes_, **kwargs)

    def find_link(
        self: VBTAssetT,
        link: tp.MaybeList[str],
        mode: str = "end",
        single_item: bool = True,
        consolidate: bool = True,
        **kwargs,
    ) -> tp.Union[VBTAssetT, tp.Any]:
        """Find item(s) corresponding to link(s)."""

        def _extend_link(link):
            from urllib.parse import urlparse

            if not urlparse(link).fragment:
                if link.endswith("/"):
                    return [link, link[:-1]]
                return [link, link + "/"]
            return [link]

        links = link
        if mode.lower() in ("exact", "end"):
            if isinstance(link, str):
                links = _extend_link(link)
            elif isinstance(link, list):
                from itertools import chain

                links = list(chain(*map(_extend_link, link)))
            else:
                raise TypeError("Link must be either string or list")
        found = self.find(links, path="link", mode=mode, single_item=single_item, **kwargs)
        if isinstance(found, (type(self), list)):
            if len(found) == 0:
                raise ValueError(f"No item matching '{link}'")
            if single_item and len(found) > 1:
                if consolidate:
                    top_parents = self.get_top_parent_links(list(found))
                    if len(top_parents) == 1:
                        for i, d in enumerate(found):
                            if d["link"] == top_parents[0]:
                                if isinstance(found, type(self)):
                                    return found.replace(data=[d], single_item=True)
                                return d
                links_block = "\n".join([d["link"] for d in found])
                raise ValueError(f"Multiple items matching '{link}':\n\n{links_block}")
        return found

    def minimize_links(self: VBTAssetT) -> VBTAssetT:
        """Minimize links."""
        return self.find_replace(
            {
                r"(https://vectorbt\.pro/pvt_[a-zA-Z0-9]+)": "$pvt_site",
                r"(https://vectorbt\.pro)": "$pub_site",
                r"(https://discord\.com/channels/[0-9]+)": "$discord",
                r"(https://github\.com/polakowo/vectorbt\.pro)": "$github",
            },
            mode="regex",
        )

    def minimize(self: VBTAssetT, minimize_links: tp.Optional[bool] = None) -> VBTAssetT:
        """Minimize by keeping the most useful information.'

        If `minimize_links` is True, replaces redundant URL prefixes by templates that can
        be easily substituted later."""
        minimize_links = self.resolve_setting(minimize_links, "minimize_links")

        new_instance = self.find_remove_empty()
        if minimize_links:
            return new_instance.minimize_links()
        return new_instance

    def select_previous(self: VBTAssetT, link: str, **kwargs) -> VBTAssetT:
        """Select the previous data item."""
        d = self.find_link(link, wrap=False, **kwargs)
        d_index = self.index(d)
        new_data = []
        if d_index > 0:
            new_data.append(self.data[d_index - 1])
        return self.replace(data=new_data, single_item=True)

    def select_next(self: VBTAssetT, link: str, **kwargs) -> VBTAssetT:
        """Select the next data item."""
        d = self.find_link(link, wrap=False, **kwargs)
        d_index = self.index(d)
        new_data = []
        if d_index < len(self.data) - 1:
            new_data.append(self.data[d_index + 1])
        return self.replace(data=new_data, single_item=True)

    def to_markdown(
        self: VBTAssetT,
        root_metadata_key: tp.Optional[tp.Key] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[VBTAssetT, tp.Any]:
        """Convert to Markdown.

        Uses `VBTAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc`.

        Use `root_metadata_key` to provide the root key for the metadata markdown.

        If `clear_metadata` is True, removes empty fields from the metadata. Arguments in
        `clear_metadata_kwargs` are passed to `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`,
        while `dump_metadata_kwargs` are passed to `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc`."""
        return self.apply(
            "to_markdown",
            root_metadata_key=root_metadata_key,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            **kwargs,
        )

    @classmethod
    def links_to_paths(
        cls,
        urls: tp.Iterable[str],
        extension: tp.Optional[str] = None,
        allow_fragments: bool = True,
    ) -> tp.List[Path]:
        """Convert links to corresponding paths."""
        from urllib.parse import urlparse

        url_paths = []
        for url in urls:
            parsed = urlparse(url, allow_fragments=allow_fragments)
            path_parts = [parsed.netloc]
            url_path = parsed.path.strip("/")
            if url_path:
                parts = url_path.split("/")
                if parsed.fragment:
                    path_parts.extend(parts)
                    if extension is not None:
                        file_name = parsed.fragment + "." + extension
                    else:
                        file_name = parsed.fragment
                    path_parts.append(file_name)
                else:
                    if len(parts) > 1:
                        path_parts.extend(parts[:-1])
                    last_part = parts[-1]
                    if extension is not None:
                        file_name = last_part + "." + extension
                    else:
                        file_name = last_part
                    path_parts.append(file_name)
            else:
                if parsed.fragment:
                    if extension is not None:
                        file_name = parsed.fragment + "." + extension
                    else:
                        file_name = parsed.fragment
                    path_parts.append(file_name)
                else:
                    if extension is not None:
                        path_parts.append("index." + extension)
                    else:
                        path_parts.append("index")
            url_paths.append(Path(os.path.join(*path_parts)))
        return url_paths

    def save_to_markdown(
        self,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: bool = False,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Path:
        """Save to Markdown files.

        If `cache` is True, uses the cache directory. Otherwise, creates a temporary directory.
        If `clear_cache` is True, deletes any existing directory before creating a new one.
        Returns the path of the directory where Markdown files are stored.

        Keyword arguments are passed to `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc`.

        Last keyword arguments in `kwargs` are forwarded down to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc.to_markdown`."""
        import tempfile
        from vectorbtpro.utils.knowledge.custom_asset_funcs import ToMarkdownAssetFunc

        cache = self.resolve_setting(cache, "cache")
        cache_dir_none = cache_dir is None
        cache_dir = self.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = self.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        if cache:
            markdown_dir = Path(cache_dir)
            if cache_dir_none:
                markdown_dir /= "markdown"
            if markdown_dir.exists():
                if clear_cache:
                    remove_dir(markdown_dir, missing_ok=True, with_contents=True)
            check_mkdir(markdown_dir, **cache_mkdir_kwargs)
        else:
            markdown_dir = Path(tempfile.mkdtemp(prefix=get_caller_qualname() + "_"))
        link_map = {d["link"]: dict(d) for d in self.data}
        url_paths = self.links_to_paths(link_map.keys(), extension="md")
        url_file_map = dict(zip(link_map.keys(), [markdown_dir / p for p in url_paths]))
        _, kwargs = ToMarkdownAssetFunc.prepare(**kwargs)

        if show_progress is None:
            show_progress = not self.single_item
        prefix = get_caller_qualname().split(".")[-1]
        pbar_kwargs = flat_merge_dicts(
            dict(
                bar_id=get_caller_qualname(),
                prefix=prefix,
            ),
            pbar_kwargs,
        )
        with ProgressBar(total=len(self.data), show_progress=show_progress, **pbar_kwargs) as pbar:
            for d in self.data:
                if not url_file_map[d["link"]].exists():
                    markdown_content = ToMarkdownAssetFunc.call(d, **kwargs)
                    check_mkdir(url_file_map[d["link"]].parent, mkdir=True)
                    with open(url_file_map[d["link"]], "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                pbar.update()

        return markdown_dir

    def to_html(
        self: VBTAssetT,
        root_metadata_key: tp.Optional[tp.Key] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        format_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[VBTAssetT, tp.Any]:
        """Convert to HTML.

        Uses `VBTAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc`.

        Arguments in `format_html_kwargs` are passed to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.format_html`.

        Last keyword arguments in `kwargs` are forwarded down to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.to_html`.

        For other arguments, see `VBTAsset.to_markdown`."""
        return self.apply(
            "to_html",
            root_metadata_key=root_metadata_key,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            to_markdown_kwargs=to_markdown_kwargs,
            format_html_kwargs=format_html_kwargs,
            **kwargs,
        )

    @classmethod
    def get_top_parent_links(cls, data: tp.List[tp.Any]) -> tp.List[str]:
        """Get links of top parents in data."""
        link_map = {d["link"]: dict(d) for d in data}
        top_parents = []
        for d in data:
            if d.get("parent", None) is None or d["parent"] not in link_map:
                top_parents.append(d["link"])
        return top_parents

    @property
    def top_parent_links(self) -> tp.List[str]:
        """Get links of top parents."""
        return self.get_top_parent_links(self.data)

    @classmethod
    def replace_urls_in_html(cls, html: str, url_map: dict) -> str:
        """Replace URLs in <a href="..."> attributes based on a provided mapping."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("bs4")

        from bs4 import BeautifulSoup
        from urllib.parse import urlparse, urlunparse

        soup = BeautifulSoup(html, "html.parser")

        for a_tag in soup.find_all("a", href=True):
            original_href = a_tag["href"]
            if original_href in url_map:
                a_tag["href"] = url_map[original_href]
            else:
                try:
                    parsed_href = urlparse(original_href)
                    base_url = urlunparse(parsed_href._replace(fragment=""))
                    if base_url in url_map:
                        new_base_url = url_map[base_url]
                        new_parsed = urlparse(new_base_url)
                        new_parsed = new_parsed._replace(fragment=parsed_href.fragment)
                        new_href = urlunparse(new_parsed)
                        a_tag["href"] = new_href
                except ValueError:
                    pass
        return str(soup)

    def save_to_html(
        self,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: bool = False,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        return_url_map: bool = False,
        **kwargs,
    ) -> tp.Union[Path, tp.Tuple[Path, dict]]:
        """Save to HTML files.

        Opens the web browser. Also, returns the path of the directory where HTML files are stored,
        and if `return_url_map` is True also returns the link->file map.

        In addition, if there are multiple top-level parents, creates an index page.

        If `cache` is True, uses the cache directory. Otherwise, creates a temporary directory.
        If `clear_cache` is True, deletes any existing directory before creating a new one.

        Keyword arguments are passed to `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc`."""
        import tempfile
        from vectorbtpro.utils.knowledge.custom_asset_funcs import ToHTMLAssetFunc

        cache = self.resolve_setting(cache, "cache")
        cache_dir_none = cache_dir is None
        cache_dir = self.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = self.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        if cache:
            html_dir = Path(cache_dir)
            if cache_dir_none:
                html_dir /= "html"
            if html_dir.exists():
                if clear_cache:
                    remove_dir(html_dir, missing_ok=True, with_contents=True)
            check_mkdir(html_dir, **cache_mkdir_kwargs)
        else:
            html_dir = Path(tempfile.mkdtemp(prefix=get_caller_qualname() + "_"))
        link_map = {d["link"]: dict(d) for d in self.data}
        top_parents = self.top_parent_links
        if len(top_parents) > 1:
            link_map["/"] = {}
        url_paths = self.links_to_paths(link_map.keys(), extension="html")
        url_file_map = dict(zip(link_map.keys(), [html_dir / p for p in url_paths]))
        url_map = {k: "file://" + str(v.resolve()) for k, v in url_file_map.items()}
        _, kwargs = ToHTMLAssetFunc.prepare(**kwargs)

        if len(top_parents) > 1:
            entry_link = "/"
            if not url_file_map[entry_link].exists():
                html = ToHTMLAssetFunc.call([link_map[link] for link in top_parents], **kwargs)
                html = self.replace_urls_in_html(html, url_map)
                check_mkdir(url_file_map[entry_link].parent, mkdir=True)
                with open(url_file_map[entry_link], "w", encoding="utf-8") as f:
                    f.write(html)

        if show_progress is None:
            show_progress = not self.single_item
        prefix = get_caller_qualname().split(".")[-1]
        pbar_kwargs = flat_merge_dicts(
            dict(
                bar_id=get_caller_qualname(),
                prefix=prefix,
            ),
            pbar_kwargs,
        )
        with ProgressBar(total=len(self.data), show_progress=show_progress, **pbar_kwargs) as pbar:
            for d in self.data:
                if not url_file_map[d["link"]].exists():
                    html = ToHTMLAssetFunc.call(d, **kwargs)
                    html = self.replace_urls_in_html(html, url_map)
                    check_mkdir(url_file_map[d["link"]].parent, mkdir=True)
                    with open(url_file_map[d["link"]], "w", encoding="utf-8") as f:
                        f.write(html)
                pbar.update()

        if return_url_map:
            return html_dir, url_map
        return html_dir

    def browse(
        self,
        entry_link: tp.Optional[str] = None,
        find_kwargs: tp.KwargsLike = None,
        open_browser: tp.Optional[bool] = None,
        **kwargs,
    ) -> Path:
        """Browse one or more HTML pages.

        Opens the web browser. Also, returns the path of the directory where HTML files are stored.

        Use `entry_link` to specify the link of the page that should be displayed first.
        If `entry_link` is None and there are multiple top-level parents, displays them as an index.
        If it's not None, it will be matched using `VBTAsset.find_link` and `find_kwargs`.

        Keyword arguments are passed to `PagesAsset.save_to_html`."""
        open_browser = self.resolve_setting(open_browser, "open_browser")

        if entry_link is None:
            if len(self.data) == 1:
                entry_link = self.data[0]["link"]
            else:
                top_parents = self.top_parent_links
                if len(top_parents) == 1:
                    entry_link = top_parents[0]
                else:
                    entry_link = "/"
        else:
            if find_kwargs is None:
                find_kwargs = {}
            d = self.find_link(entry_link, wrap=False, **find_kwargs)
            entry_link = d["link"]
        html_dir, url_map = self.save_to_html(return_url_map=True, **kwargs)
        if open_browser:
            import webbrowser

            webbrowser.open(url_map[entry_link])
        return html_dir

    def display(
        self,
        link: tp.Optional[str] = None,
        find_kwargs: tp.KwargsLike = None,
        open_browser: tp.Optional[bool] = None,
        **kwargs,
    ) -> Path:
        """Display as an HTML page.

        Opens the web browser. Also, returns the path of the temporary HTML file."""
        import tempfile

        open_browser = self.resolve_setting(open_browser, "open_browser")

        if link is not None:
            if find_kwargs is None:
                find_kwargs = {}
            single_instance = self.find_link(link, **find_kwargs)
        else:
            if len(self.data) != 1:
                raise ValueError("Must provide link")
            single_instance = self
        html = single_instance.to_html(wrap=False, single_item=True, **kwargs)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            prefix=get_caller_qualname() + "_",
            suffix=".html",
            delete=False,
        ) as f:
            f.write(html)
            file_path = Path(f.name)
        if open_browser:
            import webbrowser

            webbrowser.open("file://" + str(file_path.resolve()))
        return file_path


PagesAssetT = tp.TypeVar("PagesAssetT", bound="PagesAsset")


class PagesAsset(VBTAsset):
    """Class for working with website pages.

    For defaults, see `assets.pages` in `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge.assets.pages"

    def minimize(self: PagesAssetT, minimize_links: tp.Optional[bool] = None) -> PagesAssetT:
        new_instance = VBTAsset.minimize(self, minimize_links=minimize_links)
        new_instance = new_instance.remove(
            [
                "parent",
                "children",
                "type",
                "icon",
                "tags",
            ],
            skip_missing=True,
        )
        return new_instance

    def find_page(
        self: PagesAssetT,
        link: tp.MaybeList[str],
        aggregate: bool = False,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[PagesAssetT, tp.Any]:
        """Find the page(s) corresponding to link(s).

        Keyword arguments are passed to `VBTAsset.find_link`."""
        found = self.find_link(link, **kwargs)
        if not isinstance(found, (type(self), list)):
            return found
        if aggregate:
            if aggregate_kwargs is None:
                aggregate_kwargs = {}
            for i, d in enumerate(found):
                descendant_headings = self.select_descendant_headings(d["link"], include_link=True)
                descendant_headings = descendant_headings.aggregate(**aggregate_kwargs)
                found[i] = descendant_headings.find_link(d["link"], wrap=False)
        return found

    def find_obj(
        self,
        obj: tp.Any,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        **kwargs,
    ) -> PagesAssetT:
        """Find the page corresponding an (internal) object or reference name."""
        refname = prepare_refname(obj, module=module, resolve=resolve)
        return self.find_page(f"#({re.escape(refname)})$", mode="regex", **kwargs)

    def browse(
        self,
        entry_link: tp.Optional[str] = None,
        descendants_only: bool = False,
        aggregate: bool = False,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Path:
        new_instance = self
        if entry_link is not None and entry_link != "/" and descendants_only:
            new_instance = new_instance.select_descendants(entry_link, include_link=True)
        if aggregate:
            if aggregate_kwargs is None:
                aggregate_kwargs = {}
            new_instance = new_instance.aggregate(**aggregate_kwargs)
        return VBTAsset.browse(new_instance, entry_link=entry_link, **kwargs)

    def display(
        self,
        link: tp.Optional[str] = None,
        aggregate: bool = False,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Path:
        new_instance = self
        if link is not None:
            new_instance = new_instance.find_page(
                link,
                aggregate=aggregate,
                aggregate_kwargs=aggregate_kwargs,
            )
        elif aggregate:
            if aggregate_kwargs is None:
                aggregate_kwargs = {}
            new_instance = new_instance.aggregate(**aggregate_kwargs)
        return VBTAsset.display(new_instance, **kwargs)

    def aggregate(
        self: PagesAssetT,
        append_obj_type: tp.Optional[bool] = None,
        append_github_link: tp.Optional[bool] = None,
    ) -> PagesAssetT:
        """Aggregate pages.

        Content of each heading will be converted into markdown and concatenated into the content
        of the parent heading or page. Only regular pages and headings without parents will be left.

        If `append_obj_type` is True, will also append object type to the heading name.
        If `append_github_link` is True, will also append GitHub link to the heading name."""
        append_obj_type = self.resolve_setting(append_obj_type, "append_obj_type")
        append_github_link = self.resolve_setting(append_github_link, "append_github_link")

        link_map = {d["link"]: dict(d) for d in self.data}
        top_parents = self.top_parent_links
        aggregated_links = set()

        def _aggregate_content(link):
            node = link_map[link]
            content = node["content"]
            if content is None:
                content = ""
            if node["type"].startswith("heading"):
                level = int(node["type"].split(" ")[1])
                heading_markdown = "#" * level + " " + node["name"]
                if append_obj_type and node.get("obj_type", None) is not None:
                    heading_markdown += f" | {node['obj_type']}"
                if append_github_link and node.get("github_link", None) is not None:
                    heading_markdown += f" | [source]({node['github_link']})"
                if content == "":
                    content = heading_markdown
                else:
                    content = f"{heading_markdown}\n\n{content}"

            children = list(node["children"])
            for child in list(children):
                if child in link_map:
                    child_node = link_map[child]
                    child_content = _aggregate_content(child)
                    if child_node["type"].startswith("heading"):
                        if child_content.startswith("# "):
                            content = child_content
                        else:
                            content += f"\n\n{child_content}"
                        children.remove(child)
                        aggregated_links.add(child)

            if content != "":
                node["content"] = content
            node["children"] = children
            return content

        for top_parent in top_parents:
            _aggregate_content(top_parent)

        new_data = [link_map[link] for link in link_map if link not in aggregated_links]
        return self.replace(data=new_data)

    def select_parent(self: PagesAssetT, link: str, include_link: bool = False, **kwargs) -> PagesAssetT:
        """Select the parent page of a link."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if include_link:
            new_data.append(d)
        if d.get("parent", None):
            if d["parent"] in link_map:
                new_data.append(link_map[d["parent"]])
        return self.replace(data=new_data, single_item=True)

    def select_children(self, link: str, include_link: bool = False, **kwargs) -> PagesAssetT:
        """Select the child pages of a link."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if include_link:
            new_data.append(d)
        if d.get("children", []):
            for child in d["children"]:
                if child in link_map:
                    new_data.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def select_siblings(self, link: str, include_link: bool = False, **kwargs) -> PagesAssetT:
        """Select the sibling pages of a link."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if include_link:
            new_data.append(d)
        if d.get("parent", None):
            if d["parent"] in link_map:
                parent_d = link_map[d["parent"]]
                if parent_d.get("children", []):
                    for child in parent_d["children"]:
                        if include_link or child != d["link"]:
                            if child in link_map:
                                new_data.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def select_descendants(self, link: str, include_link: bool = False, **kwargs) -> PagesAssetT:
        """Select all descendant pages of a link."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if include_link:
            new_data.append(d)
        descendants = set()
        stack = [d]
        while stack:
            d = stack.pop()
            children = d.get("children", [])
            for child in children:
                if child in link_map and child not in descendants:
                    descendants.add(child)
                    new_data.append(link_map[child])
                    stack.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def select_branch(self, link: str, **kwargs) -> PagesAssetT:
        """Select all descendant pages of a link including the link."""
        return self.select_descendants(link, include_link=True, **kwargs)

    def select_ancestors(self, link: str, include_link: bool = False, **kwargs) -> PagesAssetT:
        """Select all ancestor pages of a link."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if include_link:
            new_data.append(d)
        ancestors = set()
        parent = d.get("parent", None)
        while parent and parent in link_map:
            if parent in ancestors:
                break
            ancestors.add(parent)
            new_data.append(link_map[parent])
            parent = link_map[parent].get("parent", None)
        return self.replace(data=new_data, single_item=False)

    def select_parent_page(self, link: str, include_link: bool = False, **kwargs) -> PagesAssetT:
        """Select parent page."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if include_link:
            new_data.append(d)
        ancestors = set()
        parent = d.get("parent", None)
        while parent and parent in link_map:
            if parent in ancestors:
                break
            ancestors.add(parent)
            new_data.append(link_map[parent])
            if link_map[parent]["type"] == "page":
                break
            parent = link_map[parent].get("parent", None)
        return self.replace(data=new_data, single_item=False)

    def select_descendant_headings(self, link: str, include_link: bool = False, **kwargs) -> PagesAssetT:
        """Select descendant headings."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if include_link:
            new_data.append(d)
        descendants = set()
        stack = [d]
        while stack:
            d = stack.pop()
            children = d.get("children", [])
            for child in children:
                if child in link_map and child not in descendants:
                    if link_map[child]["type"].startswith("heading"):
                        descendants.add(child)
                        new_data.append(link_map[child])
                        stack.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def print_site_schema(
        self,
        append_type: bool = False,
        append_obj_type: bool = False,
        structure_fragments: bool = True,
        split_fragments: bool = True,
        **dir_tree_kwargs,
    ) -> None:
        """Print site schema.

        If `structure_fragments` is True, builds a hierarchy of fragments. Otherwise,
        displays them on the same level.

        If `split_fragments` is True, displays fragments as continuation of their parents.
        Otherwise, displays them in full length.

        Keyword arguments are split between `KnowledgeAsset.describe` and
        `vectorbtpro.utils.path_.dir_tree_from_paths`."""
        link_map = {d["link"]: dict(d) for d in self.data}
        links = []
        for link, d in link_map.items():
            if not structure_fragments:
                links.append(link)
                continue
            x = d
            link_base = None
            link_fragments = []
            while x["type"].startswith("heading") and "#" in x["link"]:
                link_parts = x["link"].split("#")
                if link_base is None:
                    link_base = link_parts[0]
                link_fragments.append("#" + link_parts[1])
                if not x.get("parent", None) or x["parent"] not in link_map:
                    if x["type"].startswith("heading"):
                        level = int(x["type"].split()[1])
                        for i in range(level - 1):
                            link_fragments.append("?")
                    break
                x = link_map[x["parent"]]
            if link_base is None:
                links.append(link)
            else:
                if split_fragments and len(link_fragments) > 1:
                    link_fragments = link_fragments[::-1]
                    new_link_fragments = [link_fragments[0]]
                    for i in range(1, len(link_fragments)):
                        link_fragment1 = link_fragments[i - 1]
                        link_fragment2 = link_fragments[i]
                        if link_fragment2.startswith(link_fragment1 + "."):
                            new_link_fragments.append("." + link_fragment2[len(link_fragment1 + ".") :])
                        else:
                            new_link_fragments.append(link_fragment2)
                    link_fragments = new_link_fragments
                links.append(link_base + "/".join(link_fragments))
        paths = self.links_to_paths(links, allow_fragments=not structure_fragments)

        path_names = []
        for i, d in enumerate(link_map.values()):
            path_name = paths[i].name
            brackets = []
            if append_type:
                brackets.append(d["type"])
            if append_obj_type and d["obj_type"]:
                brackets.append(d["obj_type"])
            if brackets:
                path_name += f" [{', '.join(brackets)}]"
            path_names.append(path_name)
        if "root_name" not in dir_tree_kwargs:
            root_name = get_common_prefix(link_map.keys())
            if not root_name:
                root_name = "/"
            dir_tree_kwargs["root_name"] = root_name
        if "sort" not in dir_tree_kwargs:
            dir_tree_kwargs["sort"] = False
        if "path_names" not in dir_tree_kwargs:
            dir_tree_kwargs["path_names"] = path_names
        if "length_limit" not in dir_tree_kwargs:
            dir_tree_kwargs["length_limit"] = None
        print(dir_tree_from_paths(paths, **dir_tree_kwargs))


MessagesAssetT = tp.TypeVar("MessagesAssetT", bound="MessagesAsset")


class MessagesAsset(VBTAsset):
    """Class for working with Discord messages.

    For defaults, see `assets.messages` in `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge.assets.messages"

    def minimize(self: MessagesAssetT, minimize_links: tp.Optional[bool] = None) -> MessagesAssetT:
        new_instance = VBTAsset.minimize(self, minimize_links=minimize_links)
        new_instance = new_instance.remove(
            [
                "block",
                "thread",
                "replies",
                "mentions",
                "reactions",
            ],
            skip_missing=True,
        )
        return new_instance

    def aggregate_messages(
        self: MessagesAssetT,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[MessagesAssetT, tp.Any]:
        """Aggregate attachments by message.

        Argument `metadata_format` can be either "markdown" or "html". For keyword arguments, see
        `MessagesAsset.to_markdown` and `MessagesAsset.to_html` respectively.

        Uses `MessagesAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.AggMessageAssetFunc`."""
        return self.apply(
            "agg_message",
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            **kwargs,
        )

    def aggregate_blocks(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[MessagesAssetT, tp.Any]:
        """Aggregate messages by block.

        First, uses `MessagesAsset.reduce` on `vectorbtpro.utils.knowledge.base_asset_funcs.CollectAssetFunc`
        to collect data items by the field "block". Keyword arguments in `collect_kwargs` are passed here.
        Argument `uniform_groups` is True by default. Then, uses `MessagesAsset.apply` on
        `vectorbtpro.utils.knowledge.custom_asset_funcs.AggBlockAssetFunc` to aggregate each collected data item.

        Use `aggregate_fields` to provide a set of fields to be aggregated rather than used in child metadata.
        It can be True to aggregate all lists and False to aggregate none.

        If `parent_links_only` is True, doesn't include links in the metadata of each message.

        Argument `metadata_format` can be either "markdown" or "html". For other keyword arguments, see
        `MessagesAsset.to_markdown` and `MessagesAsset.to_html` respectively."""
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by="block", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_block",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    def aggregate_threads(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[MessagesAssetT, tp.Any]:
        """Aggregate messages by thread.

        Same as `MessagesAsset.aggregate_blocks` but for threads.

        Uses `vectorbtpro.utils.knowledge.custom_asset_funcs.AggThreadAssetFunc`."""
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by="thread", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_thread",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    def aggregate_channels(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[MessagesAssetT, tp.Any]:
        """Aggregate messages by channel.

        Same as `MessagesAsset.aggregate_threads` but for channels.

        Uses `vectorbtpro.utils.knowledge.custom_asset_funcs.AggChannelAssetFunc`."""
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by="channel", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_channel",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    @property
    def lowest_aggregate_by(self) -> str:
        """Get the lowest level that aggregates all messages."""
        if len(self) == 1 and self[0].get("attachments", []):
            return "message"
        try:
            if len(set(self.get("block"))) == 1:
                return "block"
        except KeyError:
            pass
        try:
            if len(set(self.get("thread"))) == 1:
                return "thread"
        except KeyError:
            pass
        try:
            if len(set(self.get("channel"))) == 1:
                return "channel"
        except KeyError:
            pass
        raise ValueError("Must provide by")

    def aggregate(self, by: tp.Optional[str] = None, *args, **kwargs) -> tp.Union[MessagesAssetT, tp.Any]:
        """Aggregate by "message" (attachments), "block", "thread", or "channel".

        If `by` is None, uses `MessagesAsset.lowest_aggregate_by`."""
        if by is None:
            by = self.lowest_aggregate_by
        if not by.lower().endswith("s"):
            by += "s"
        return getattr(self, "aggregate_" + by.lower())(*args, **kwargs)

    def select_reference(self: MessagesAssetT, link: str, **kwargs) -> MessagesAssetT:
        """Select the reference message."""
        d = self.find_link(link, wrap=False, **kwargs)
        reference = d.get("reference", None)
        new_data = []
        if reference:
            for d2 in self.data:
                if d2["reference"] == reference:
                    new_data.append(d2)
                    break
        return self.replace(data=new_data, single_item=True)

    def select_replies(self: MessagesAssetT, link: str, **kwargs) -> MessagesAssetT:
        """Select the reply messages."""
        d = self.find_link(link, wrap=False, **kwargs)
        replies = d.get("replies", [])
        new_data = []
        if replies:
            reply_data = {reply: None for reply in replies}
            replies_found = 0
            for d2 in self.data:
                if d2["link"] in reply_data:
                    reply_data[d2["link"]] = d2
                    replies_found += 1
                    if replies_found == len(replies):
                        break
            new_data = list(reply_data.values())
        return self.replace(data=new_data, single_item=True)

    def select_block(self: MessagesAssetT, link: str, include_link: bool = True, **kwargs) -> MessagesAssetT:
        """Select the messages that belong to the block of a link."""
        d = self.find_link(link, wrap=False, **kwargs)
        new_data = []
        for d2 in self.data:
            if d2["block"] == d["block"] and (include_link or d2["link"] != d["link"]):
                new_data.append(d2)
        return self.replace(data=new_data, single_item=False)

    def select_thread(self: MessagesAssetT, link: str, include_link: bool = True, **kwargs) -> MessagesAssetT:
        """Select the messages that belong to the thread of a link."""
        d = self.find_link(link, wrap=False, **kwargs)
        new_data = []
        for d2 in self.data:
            if d2["thread"] == d["thread"] and (include_link or d2["link"] != d["link"]):
                new_data.append(d2)
        return self.replace(data=new_data, single_item=False)

    def select_channel(self: MessagesAssetT, link: str, include_link: bool = True, **kwargs) -> MessagesAssetT:
        """Select the messages that belong to the channel of a link."""
        d = self.find_link(link, wrap=False, **kwargs)
        new_data = []
        for d2 in self.data:
            if d2["channel"] == d["channel"] and (include_link or d2["link"] != d["link"]):
                new_data.append(d2)
        return self.replace(data=new_data, single_item=False)
