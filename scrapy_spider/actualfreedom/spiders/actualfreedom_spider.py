import scrapy

class ActualFreedomSpider(scrapy.Spider):
    name = 'actualfreedom'
    allowed_domains = ['actualfreedom.com.au']
    start_urls = ['http://www.actualfreedom.com.au/']

    def parse(self, response):
        # Extract text from the current page
        page_text = response.xpath("//body//text()").getall()
        page_text = ' '.join(page_text).strip()

        yield {
            'url': response.url,
            'text': page_text
        }

        # Follow all links on the current page, excluding JavaScript links
        for href in response.css('a::attr(href)').getall():
            if not href.startswith('javascript:'):  # Skip JavaScript links
                full_url = response.urljoin(href)
                yield scrapy.Request(full_url, callback=self.parse)