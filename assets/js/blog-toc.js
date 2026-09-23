(() => {
  const article = document.querySelector('.technical-blog .post-content');
  const nav = document.querySelector('.blog-toc');
  if (!article || !nav) return;
  const headings = [...article.querySelectorAll('h2')];
  if (!headings.length) return;
  const label = document.createElement('p');
  label.className = 'blog-toc-label';
  label.textContent = 'On this page';
  const list = document.createElement('ul');
  const links = headings.map((heading, index) => {
    if (!heading.id) heading.id = `section-${index + 1}`;
    const item = document.createElement('li');
    const link = document.createElement('a');
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent.replace(/\s+/g, ' ').trim();
    item.append(link);
    list.append(item);
    return link;
  });
  nav.append(label, list);
  nav.hidden = false;
  const narrow = window.matchMedia('(max-width: 1100px)');
  const placeNavigation = () => {
    if (narrow.matches) document.querySelector('.post-header').after(nav);
    else document.querySelector('.blog-layout').prepend(nav);
  };
  narrow.addEventListener('change', placeNavigation);
  placeNavigation();
  let queued = false;
  let current = -1;
  const update = () => {
    queued = false;
    let active = 0;
    headings.forEach((heading, index) => {
      if (heading.getBoundingClientRect().top <= 120) active = index;
    });
    if (current === active) return;
    links.forEach((link, index) => {
      if (index === active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
    current = active;
  };
  window.addEventListener('scroll', () => {
    if (!queued) { queued = true; requestAnimationFrame(update); }
  }, { passive: true });
  window.addEventListener('resize', update);
  update();
  article.querySelectorAll('table').forEach((table, index) => {
    if (table.parentElement.classList.contains('blog-table-scroll')) return;
    const wrapper = document.createElement('div');
    wrapper.className = 'blog-table-scroll';
    wrapper.tabIndex = 0;
    wrapper.setAttribute('role', 'region');
    wrapper.setAttribute('aria-label', `Scrollable data table ${index + 1}`);
    table.before(wrapper);
    wrapper.append(table);
  });
})();
