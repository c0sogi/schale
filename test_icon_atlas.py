"""Test icon atlas preparation."""

from schale.scanner._icons import IconAtlas

print("Preparing icon atlas...")
atlas = IconAtlas()
atlas.prepare()

print(f"\nAtlas prepared: {atlas._prepared}")
print(f"Total templates loaded: {len(atlas._templates)}")

if atlas._templates:
    print(f"\nFirst 10 templates:")
    for i, (name, template) in enumerate(list(atlas._templates.items())[:10]):
        print(f"  {i+1}. {name} (T{template.tier}, category={template.category}, blueprint={template.is_blueprint})")

    # Count by category
    from collections import Counter
    categories = Counter(t.category for t in atlas._templates.values())
    print(f"\nTemplates by category:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")
else:
    print("\nWARNING: No templates loaded!")
