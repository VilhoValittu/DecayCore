{% for nav_path in site.header_pages %}
  {% assign nav_page = site.pages | where: "path", nav_path | first %}
  {% if nav_page.title %}
    <a href="{{ nav_page.url | relative_url }}"{% if page.url == nav_page.url %} aria-current="page"{% endif %}>{{ nav_page.nav_title | default: nav_page.title }}</a>
  {% endif %}
{% endfor %}
<a href="https://github.com/VilhoValittu/DecayCore/releases">Releases</a>
