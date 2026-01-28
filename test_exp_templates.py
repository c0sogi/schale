"""Check if Exp templates are loaded."""

from schale.scanner._icons import IconAtlas

atlas = IconAtlas()
atlas.prepare()

print(f"Total templates: {len(atlas._templates)}")

exp_templates = [name for name in atlas._templates.keys() if 'exp' in name.lower()]
print(f"\nExp-related templates ({len(exp_templates)}):")
for name in sorted(exp_templates)[:20]:
    tmpl = atlas._templates[name]
    print(f"  {name}: {tmpl.category} T{tmpl.tier}, blueprint={tmpl.is_blueprint}")
