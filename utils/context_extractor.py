# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/3/31
# Description: 网页内容提取

import math
from typing import List
from bs4 import BeautifulSoup
import bs4


class NodeInfo(object):
    def __init__(self):
        self.textCnt = 0.0
        self.linkTextCnt = 0.0
        self.tagCnt = 0.0
        self.linkTagCnt = 0.0
        self.density = 0.0
        self.densitySum = 0.0
        self.score = 0.0
        self.pCount = 0.0
        self.leafList = list()


class ContentExtractor(object):
    def __init__(self, page_src: str):
        self.soup = BeautifulSoup(page_src, "html5lib")
        self.infoMap = dict()

    def cleanTag(self):
        # replace br with \n
        # for brTag in self.soup.find_all("br"):
        #     brTag.repalceWith("\n")

        # remove following tag
        targetTagNames = ["script", "noscript", "style", "iframe"]
        for target in targetTagNames:
            for ele in self.soup.find_all(target):
                ele.decompose()

        for e in self.soup.descendants:
            if isinstance(e, bs4.Comment):
                e.extract()

    def getContent(self):
        self.cleanTag()
        self.compute_info(self.soup.body)
        maxScore = 0
        contentTag = None
        for tag in self.infoMap.keys():
            if tag.name == "a" or tag.name == "body":
                continue
            score = self.compute_score(tag)
            if score > maxScore:
                maxScore = score
                contentTag = tag

        if contentTag:
            contentTag.prettify()
            for p in contentTag.find_all("p"):
                br = self.soup.new_string("\n")
                p.insert_after(br)
            contentTag.prettify()
            return contentTag.text

    def compute_info(self, node: bs4.PageElement):
        if isinstance(node, bs4.Comment):
            return NodeInfo()
        elif isinstance(node, bs4.Tag):
            info = NodeInfo()
            for child in node.children:
                childInfo = self.compute_info(child)
                info.textCnt += childInfo.textCnt
                info.linkTextCnt += childInfo.linkTextCnt
                info.tagCnt += childInfo.tagCnt
                info.linkTagCnt += childInfo.linkTagCnt
                info.leafList.extend(childInfo.leafList)
                info.densitySum += childInfo.density
                info.pCount += childInfo.pCount

            info.tagCnt += 1

            if node.name == "a":
                info.linkTextCnt = info.textCnt
                info.linkTagCnt += 1
            elif node.name == "p":
                info.pCount += 1

            pureLen = info.textCnt - info.linkTextCnt
            tag_len = info.tagCnt - info.linkTagCnt
            if pureLen == 0 or tag_len == 0:
                info.density = 0
            else:
                info.density = pureLen / tag_len

            self.infoMap[node] = info
            return info
        elif isinstance(node, bs4.NavigableString):
            info = NodeInfo()
            info.textCnt = len(node)
            info.leafList.append(info.textCnt)
            return info
        else:
            raise Exception("???????")

    def variance(self, vals: List[int]):
        if len(vals) == 0:
            return 0

        if len(vals) == 1:
            return vals[0] / 2
        total = sum(vals)
        avg = total / len(vals)
        total = 0
        for val in vals:
            total += (val - avg) * (val - avg)
        total /= len(vals)
        return total

    def compute_score(self, node: bs4.PageElement):
        info = self.infoMap.get(node)
        a = math.sqrt(self.variance(info.leafList) + 1)
        score = math.log(a) * info.densitySum * math.log(info.textCnt - info.linkTextCnt + 1) * math.log(info.pCount + 2)
        return score
