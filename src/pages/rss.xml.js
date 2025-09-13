import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const posts = await getCollection('blog');
  
  return rss({
    title: 'AI Edu-Blog',
    description: 'Latest insights on AI, machine learning, and automation technologies',
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.publishDate,
      description: post.data.description,
      author: post.data.author,
      link: `/blog/${post.slug}/`,
    })),
    customData: `<language>id-id</language>`,
  });
}