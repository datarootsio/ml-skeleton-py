# -*- coding: utf-8 -*-
import scrapy
from scrapy import Selector

class RSSFeedSpider(scrapy.Spider):
    name = 'rss'
    categories = ['binnenland']
    url_format = 'https://www.vrt.be/vrtnws/nl.rss.{}.xml'


    def start_requests(self):
        for c in self.categories:
            yield scrapy.Request(url=self.url_format.format(c), callback=self.parse)


    def parse(self, response):
        body = Selector(text=response.body)
        rss_update = body.xpath('//feed/updated/text()').extract_first()
        rss_cat = body.xpath('//feed/title/text()').extract_first()
        for entry in body.xpath('//entry'):
            yield {
                'rss_update': rss_update,
                'rss_category': rss_cat,
                'title': entry.xpath('./title/text()').extract_first(),
                'id': entry.xpath('./id/text()').extract_first(),
                'published': entry.xpath('./published/text()').extract_first(),
                'updated': entry.xpath('./updated/text()').extract_first(),
                'summary': entry.xpath('./summary/text()').extract_first(),
                # 'tag': entry.xpath('./vrtns:nstag/text()').extract_first(),
                'links': entry.xpath('./link/@href').extract(),
            }