import sys
import os
from typing import Iterator, Union, Tuple, Optional, Any
from .lattice import Lattice, Node, SurfaceNode, BOS, EOS, NodeType
from .dic import UserDictionary, CompiledUserDictionary
from .system_dic import SystemDictionary, MMapSystemDictionary
from .fst import Matcher

try:
    from janome.sysdic import all_fstdata, connections
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from sysdic import all_fstdata, connections
DEFAULT_MMAP_MODE = sys.maxsize > 2**32


class Token(object):
    def __init__(self, node: Node, extra: Optional[Tuple] = None):
        self.node = node
        self.extra = extra

    def __getattr__(self, name) -> Any:
        if name == "surface":
            return self.node.surface
        elif name == "part_of_speech":
            return self.extra[0] if self.extra else self.node.part_of_speech
        elif name == "infl_type":
            return self.extra[1] if self.extra else self.node.infl_type
        elif name == "infl_form":
            return self.extra[2] if self.extra else self.node.infl_form
        elif name == "base_form":
            return self.extra[3] if self.extra else self.node.base_form
        elif name == "reading":
            return self.extra[4] if self.extra else self.node.reading
        elif name == "phonetic":
            return self.extra[5] if self.extra else self.node.phonetic
        elif name == "node_type":
            return self.node.node_type
        else:
            None

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def __str__(self):
        return (
            f"{self.surface}\t"
            f"{self.part_of_speech},{self.infl_type},{self.infl_form},{self.base_form},{self.reading},{self.phonetic}"
        )


class Tokenizer(object):
    MAX_CHUNK_SIZE = 1024
    CHUNK_SIZE = 500

    def __init__(
        self,
        udic: str = "",
        *,
        udic_enc: str = "utf8",
        udic_type: str = "ipadic",
        max_unknown_length: int = 1024,
        wakati: bool = False,
        mmap: bool = DEFAULT_MMAP_MODE,
        dotfile: str = "",
    ):
        self.sys_dic: Union[SystemDictionary, MMapSystemDictionary]
        self.user_dic: Optional[Union[UserDictionary, CompiledUserDictionary]]
        self.wakati = wakati
        self.matcher = Matcher(all_fstdata())
        if mmap:
            self.sys_dic = MMapSystemDictionary.instance()
        else:
            self.sys_dic = SystemDictionary.instance()
        if udic:
            if udic.endswith(".csv"):
                self.user_dic = UserDictionary(udic, udic_enc, udic_type, connections)
            elif os.path.isdir(udic):
                self.user_dic = CompiledUserDictionary(udic, connections)
            else:
                self.user_dic = None
        else:
            self.user_dic = None
        self.max_unknown_length = max_unknown_length

    def tokenize(
        self, text: str, *, wakati: bool = False, baseform_unk: bool = True, dotfile: str = ""
    ) -> Iterator[Union[Token, str]]:
        if self.wakati:
            wakati = True
        if dotfile and len(text) < Tokenizer.MAX_CHUNK_SIZE:
            return self.__tokenize_stream(text, wakati, baseform_unk, dotfile)
        else:
            return self.__tokenize_stream(text, wakati, baseform_unk, "")

    def __tokenize_stream(self, text, wakati, baseform_unk, dotfile):
        text = text.strip()
        text_length = len(text)
        processed = 0
        while processed < text_length:
            tokens, pos = self.__tokenize_partial(text[processed:], wakati, baseform_unk, dotfile)
            for token in tokens:
                yield token
            processed += pos

    def __tokenize_partial(self, text, wakati, baseform_unk, dotfile):
        if self.wakati and not wakati:
            raise WakatiModeOnlyException
        chunk_size = min(len(text), Tokenizer.MAX_CHUNK_SIZE)
        lattice = Lattice(chunk_size, self.sys_dic)
        pos = 0
        while not self.__should_split(text, pos):
            encoded_partial_text = text[pos : pos + min(50, chunk_size - pos)].encode("utf-8")
            if self.user_dic:
                entries = self.user_dic.lookup(encoded_partial_text)
                for e in entries:
                    lattice.add(SurfaceNode(e, NodeType.USER_DICT))
                matched = len(entries) > 0
            entries = self.sys_dic.lookup(encoded_partial_text, self.matcher)
            for e in entries:
                lattice.add(SurfaceNode(e, NodeType.SYS_DICT))
            matched = len(entries) > 0
            cates = self.sys_dic.get_char_categories(text[pos])
            if cates:
                for cate in cates:
                    if matched and not self.sys_dic.unknown_invoked_always(cate):
                        continue
                    length = (
                        self.sys_dic.unknown_length(cate)
                        if not self.sys_dic.unknown_grouping(cate)
                        else self.max_unknown_length
                    )
                    assert length >= 0
                    buf = text[pos]
                    for p in range(pos + 1, min(chunk_size, pos + length + 1)):
                        _cates = self.sys_dic.get_char_categories(text[p])
                        if cate in _cates or any(cate in _compat_cates for _compat_cates in _cates.values()):
                            buf += text[p]
                        else:
                            break
                    unknown_entries = self.sys_dic.unknowns.get(cate)
                    assert unknown_entries
                    for entry in unknown_entries:
                        left_id, right_id, cost, part_of_speech = entry
                        base_form = buf if baseform_unk else "*"
                        dummy_dict_entry = (buf, left_id, right_id, cost, part_of_speech, "*", "*", base_form, "*", "*")
                        lattice.add(Node(dummy_dict_entry, NodeType.UNKNOWN))
            pos += lattice.forward()
        lattice.end()
        min_cost_path = lattice.backward()
        assert isinstance(min_cost_path[0], BOS)
        assert isinstance(min_cost_path[-1], EOS)
        if wakati:
            tokens = [node.surface for node in min_cost_path[1:-1]]
        else:
            tokens = []
            for node in min_cost_path[1:-1]:
                if type(node) == SurfaceNode and node.node_type == NodeType.SYS_DICT:
                    tokens.append(Token(node, self.sys_dic.lookup_extra(node.num)))
                elif type(node) == SurfaceNode and node.node_type == NodeType.USER_DICT:
                    tokens.append(Token(node, self.user_dic.lookup_extra(node.num)))
                else:
                    tokens.append(Token(node))
        if dotfile:
            lattice.generate_dotfile(filename=dotfile)
        return (tokens, pos)

    def __should_split(self, text, pos):
        return (
            pos >= len(text)
            or pos >= Tokenizer.MAX_CHUNK_SIZE
            or (pos >= Tokenizer.CHUNK_SIZE and self.__splittable(text[:pos]))
        )

    def __splittable(self, text):
        return self.__is_punct(text[-1]) or self.__is_newline(text)

    def __is_punct(self, c):
        return c == "、" or c == "。" or c == "," or c == "." or c == "？" or c == "?" or c == "！" or c == "!"

    def __is_newline(self, text):
        return text.endswith("\n\n") or text.endswith("\r\n\r\n")


class WakatiModeOnlyException(Exception):
    pass
